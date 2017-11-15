"""
Offer numeric state listening automation rules.

For more details about this automation rule, please refer to the documentation
at https://home-assistant.io/docs/automation/trigger/#numeric-state-trigger
"""
import asyncio
import logging

import voluptuous as vol

from homeassistant.core import callback
from homeassistant.const import (
    CONF_VALUE_TEMPLATE, CONF_PLATFORM, CONF_ENTITY_ID,
    CONF_BELOW, CONF_ABOVE, CONF_FOR)
from homeassistant.helpers.event import (
    async_track_state_change, async_track_same_state)
from homeassistant.helpers import condition, config_validation as cv

TRIGGER_SCHEMA = vol.All(vol.Schema({
    vol.Required(CONF_PLATFORM): 'numeric_state',
    vol.Required(CONF_ENTITY_ID): cv.entity_ids,
    vol.Optional(CONF_BELOW): vol.Coerce(float),
    vol.Optional(CONF_ABOVE): vol.Coerce(float),
    vol.Optional(CONF_VALUE_TEMPLATE): cv.template,
    vol.Optional(CONF_FOR): vol.All(cv.time_period, cv.positive_timedelta),
}), cv.has_at_least_one_key(CONF_BELOW, CONF_ABOVE))

_LOGGER = logging.getLogger(__name__)


@asyncio.coroutine
def async_trigger(hass, config, action):
    """Listen for state changes based on configuration."""
    entity_id = config.get(CONF_ENTITY_ID)
    below = config.get(CONF_BELOW)
    above = config.get(CONF_ABOVE)
    time_delta = config.get(CONF_FOR)
    value_template = config.get(CONF_VALUE_TEMPLATE)
    async_remove_track_same = None
    already_triggered = False

    if value_template is not None:
        value_template.hass = hass

    @callback
    def check_numeric_state(entity, from_s, to_s):
        """Return True if criteria are now met."""
        if to_s is None:
            return False

        variables = {
            'trigger': {
                'platform': 'numeric_state',
                'entity_id': entity,
                'below': below,
                'above': above,
            }
        }
        return condition.async_numeric_state(
            hass, to_s, below, above, value_template, variables)

    @callback
    def state_automation_listener(entity, from_s, to_s):
        """Listen for state changes and calls action."""
        nonlocal already_triggered, async_remove_track_same

        @callback
        def call_action():
            """Call action with right context."""
            hass.async_run_job(action, {
                'trigger': {
                    'platform': 'numeric_state',
                    'entity_id': entity,
                    'below': below,
                    'above': above,
                    'from_state': from_s,
                    'to_state': to_s,
                }
            })

        matching = check_numeric_state(entity, from_s, to_s)

        if matching and not already_triggered:
            if time_delta:
                async_remove_track_same = async_track_same_state(
                    hass, time_delta, call_action, entity_ids=entity_id,
                    async_check_same_func=check_numeric_state)
            else:
                call_action()

        already_triggered = matching

    unsub = async_track_state_change(
        hass, entity_id, state_automation_listener)

    @callback
    def async_remove():
        """Remove state listeners async."""
        unsub()
        if async_remove_track_same:
            async_remove_track_same()  # pylint: disable=not-callable

    return async_remove
