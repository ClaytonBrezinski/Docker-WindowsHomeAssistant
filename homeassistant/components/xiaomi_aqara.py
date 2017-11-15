"""Support for Xiaomi Gateways."""
import asyncio
import logging
import voluptuous as vol
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers import discovery
from homeassistant.helpers.entity import Entity
from homeassistant.components.discovery import SERVICE_XIAOMI_GW
from homeassistant.const import (ATTR_BATTERY_LEVEL, EVENT_HOMEASSISTANT_STOP,
                                 CONF_MAC, CONF_HOST, CONF_PORT)

REQUIREMENTS = ['PyXiaomiGateway==0.6.0']

ATTR_GW_MAC = 'gw_mac'
ATTR_RINGTONE_ID = 'ringtone_id'
ATTR_RINGTONE_VOL = 'ringtone_vol'
ATTR_DEVICE_ID = 'device_id'
CONF_DISCOVERY_RETRY = 'discovery_retry'
CONF_GATEWAYS = 'gateways'
CONF_INTERFACE = 'interface'
CONF_KEY = 'key'
DOMAIN = 'xiaomi_aqara'
PY_XIAOMI_GATEWAY = "xiaomi_gw"

SERVICE_PLAY_RINGTONE = 'play_ringtone'
SERVICE_STOP_RINGTONE = 'stop_ringtone'
SERVICE_ADD_DEVICE = 'add_device'
SERVICE_REMOVE_DEVICE = 'remove_device'


GW_MAC = vol.All(
    cv.string,
    lambda value: value.replace(':', '').lower(),
    vol.Length(min=12, max=12)
)


SERVICE_SCHEMA_PLAY_RINGTONE = vol.Schema({
    vol.Required(ATTR_RINGTONE_ID):
        vol.All(vol.Coerce(int), vol.NotIn([9, 14, 15, 16, 17, 18, 19])),
    vol.Optional(ATTR_RINGTONE_VOL):
        vol.All(vol.Coerce(int), vol.Clamp(min=0, max=100))
})

SERVICE_SCHEMA_REMOVE_DEVICE = vol.Schema({
    vol.Required(ATTR_DEVICE_ID):
        vol.All(cv.string, vol.Length(min=14, max=14))
})


GATEWAY_CONFIG = vol.Schema({
    vol.Optional(CONF_MAC, default=None): vol.Any(GW_MAC, None),
    vol.Optional(CONF_KEY, default=None):
        vol.All(cv.string, vol.Length(min=16, max=16)),
    vol.Optional(CONF_HOST): cv.string,
    vol.Optional(CONF_PORT, default=9898): cv.port,
})


def _fix_conf_defaults(config):
    """Update some config defaults."""
    config['sid'] = config.pop(CONF_MAC, None)

    if config.get(CONF_KEY) is None:
        _LOGGER.warning(
            'Key is not provided for gateway %s. Controlling the gateway '
            'will not be possible.', config['sid'])

    if config.get(CONF_HOST) is None:
        config.pop(CONF_PORT)

    return config


DEFAULT_GATEWAY_CONFIG = [{CONF_MAC: None, CONF_KEY: None}]

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Optional(CONF_GATEWAYS, default=DEFAULT_GATEWAY_CONFIG):
            vol.All(cv.ensure_list, [GATEWAY_CONFIG], [_fix_conf_defaults]),
        vol.Optional(CONF_INTERFACE, default='any'): cv.string,
        vol.Optional(CONF_DISCOVERY_RETRY, default=3): cv.positive_int
    })
}, extra=vol.ALLOW_EXTRA)

_LOGGER = logging.getLogger(__name__)


def setup(hass, config):
    """Set up the Xiaomi component."""
    gateways = []
    interface = 'any'
    discovery_retry = 3
    if DOMAIN in config:
        gateways = config[DOMAIN][CONF_GATEWAYS]
        interface = config[DOMAIN][CONF_INTERFACE]
        discovery_retry = config[DOMAIN][CONF_DISCOVERY_RETRY]

    @asyncio.coroutine
    def xiaomi_gw_discovered(service, discovery_info):
        """Called when Xiaomi Gateway device(s) has been found."""
        # We don't need to do anything here, the purpose of HA's
        # discovery service is to just trigger loading of this
        # component, and then its own discovery process kicks in.

    discovery.listen(hass, SERVICE_XIAOMI_GW, xiaomi_gw_discovered)

    from PyXiaomiGateway import PyXiaomiGateway
    xiaomi = hass.data[PY_XIAOMI_GATEWAY] = PyXiaomiGateway(
        hass.add_job, gateways, interface)

    _LOGGER.debug("Expecting %s gateways", len(gateways))
    for k in range(discovery_retry):
        _LOGGER.info('Discovering Xiaomi Gateways (Try %s)', k + 1)
        xiaomi.discover_gateways()
        if len(xiaomi.gateways) >= len(gateways):
            break

    if not xiaomi.gateways:
        _LOGGER.error("No gateway discovered")
        return False
    xiaomi.listen()
    _LOGGER.debug("Gateways discovered. Listening for broadcasts")

    for component in ['binary_sensor', 'sensor', 'switch', 'light', 'cover']:
        discovery.load_platform(hass, component, DOMAIN, {}, config)

    def stop_xiaomi(event):
        """Stop Xiaomi Socket."""
        _LOGGER.info("Shutting down Xiaomi Hub.")
        xiaomi.stop_listen()

    hass.bus.listen_once(EVENT_HOMEASSISTANT_STOP, stop_xiaomi)

    def play_ringtone_service(call):
        """Service to play ringtone through Gateway."""
        ring_id = call.data.get(ATTR_RINGTONE_ID)
        gateway = call.data.get(ATTR_GW_MAC)

        kwargs = {'mid': ring_id}

        ring_vol = call.data.get(ATTR_RINGTONE_VOL)
        if ring_vol is not None:
            kwargs['vol'] = ring_vol

        gateway.write_to_hub(gateway.sid, **kwargs)

    def stop_ringtone_service(call):
        """Service to stop playing ringtone on Gateway."""
        gateway = call.data.get(ATTR_GW_MAC)
        gateway.write_to_hub(gateway.sid, mid=10000)

    def add_device_service(call):
        """Service to add a new sub-device within the next 30 seconds."""
        gateway = call.data.get(ATTR_GW_MAC)
        gateway.write_to_hub(gateway.sid, join_permission='yes')
        hass.components.persistent_notification.async_create(
            'Join permission enabled for 30 seconds! '
            'Please press the pairing button of the new device once.',
            title='Xiaomi Aqara Gateway')

    def remove_device_service(call):
        """Service to remove a sub-device from the gateway."""
        device_id = call.data.get(ATTR_DEVICE_ID)
        gateway = call.data.get(ATTR_GW_MAC)
        gateway.write_to_hub(gateway.sid, remove_device=device_id)

    gateway_only_schema = _add_gateway_to_schema(xiaomi, vol.Schema({}))

    hass.services.async_register(
        DOMAIN, SERVICE_PLAY_RINGTONE, play_ringtone_service,
        schema=_add_gateway_to_schema(xiaomi, SERVICE_SCHEMA_PLAY_RINGTONE))

    hass.services.async_register(
        DOMAIN, SERVICE_STOP_RINGTONE, stop_ringtone_service,
        schema=gateway_only_schema)

    hass.services.async_register(
        DOMAIN, SERVICE_ADD_DEVICE, add_device_service,
        schema=gateway_only_schema)

    hass.services.async_register(
        DOMAIN, SERVICE_REMOVE_DEVICE, remove_device_service,
        schema=_add_gateway_to_schema(xiaomi, SERVICE_SCHEMA_REMOVE_DEVICE))

    return True


class XiaomiDevice(Entity):
    """Representation a base Xiaomi device."""

    def __init__(self, device, name, xiaomi_hub):
        """Initialize the xiaomi device."""
        self._state = None
        self._sid = device['sid']
        self._name = '{}_{}'.format(name, self._sid)
        self._write_to_hub = xiaomi_hub.write_to_hub
        self._get_from_hub = xiaomi_hub.get_from_hub
        self._device_state_attributes = {}
        xiaomi_hub.callbacks[self._sid].append(self.push_data)
        self.parse_data(device['data'])
        self.parse_voltage(device['data'])

    @property
    def name(self):
        """Return the name of the device."""
        return self._name

    @property
    def should_poll(self):
        """No polling needed."""
        return False

    @property
    def device_state_attributes(self):
        """Return the state attributes."""
        return self._device_state_attributes

    def push_data(self, data):
        """Push from Hub."""
        _LOGGER.debug("PUSH >> %s: %s", self, data)
        if self.parse_data(data) or self.parse_voltage(data):
            self.schedule_update_ha_state()

    def parse_voltage(self, data):
        """Parse battery level data sent by gateway."""
        if 'voltage' not in data:
            return False
        max_volt = 3300
        min_volt = 2800
        voltage = data['voltage']
        voltage = min(voltage, max_volt)
        voltage = max(voltage, min_volt)
        percent = ((voltage - min_volt) / (max_volt - min_volt)) * 100
        self._device_state_attributes[ATTR_BATTERY_LEVEL] = round(percent, 1)
        return True

    def parse_data(self, data):
        """Parse data sent by gateway."""
        raise NotImplementedError()


def _add_gateway_to_schema(xiaomi, schema):
    """Extend a voluptuous schema with a gateway validator."""
    def gateway(sid):
        """Convert sid to a gateway."""
        sid = str(sid).replace(':', '').lower()

        for gateway in xiaomi.gateways.values():
            if gateway.sid == sid:
                return gateway

        raise vol.Invalid('Unknown gateway sid {}'.format(sid))

    gateways = list(xiaomi.gateways.values())
    kwargs = {}

    # If the user has only 1 gateway, make it the default for services.
    if len(gateways) == 1:
        kwargs['default'] = gateways[0]

    return schema.extend({
        vol.Required(ATTR_GW_MAC, **kwargs): gateway
    })
