import numpy as nup
import pandas as pand
import calmap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

if __name__ == "__main__":

    DB_URL = "sqlite:///./home-assistant_v2.db"
    engine = create_engine(DB_URL)

    # gather entities from the database
    entities = engine.execute("""SELECT entity_id, COUNT(*) FROM states WHERE commsDevice = "Clayton's phone"  GROUP BY entity_id""")

    # get rows from query into a pandas dataframe and name columns
    devicesData = pand.DataFrame(devicestatesquery.fetchall())
    devicesData.columns = ['device', 'lastChanged', 'futureChange']

    # create a dataframe with an index ranging from the min to the max\
    # last_changed timestamp from our query
    dates = pand.DataFrame(index=pand.date_range(min(pand.to_datetime(devicesData['lastChanged']).dt.date.unique()),
                                                 max(pand.to_datetime(devicesData['lastChanged']).dt.date.unique()
                                                     ) - timedelta(days=1)))

    # create a last_changed_date column with only the date portion of \
    # last_changed value
    devicesData['lastChangedDate'] = pand.to_datetime(devicesData['lastChanged']).apply(lambda x: x.date())

    # create a HomeTime column with a zero value
    dates['HomeTime'] = timedelta(days=0)

    # grab the index values of our dates dataframe
    dateList = pand.to_datetime(dates.index.values.tolist())

    # process for each date within the list
    for date in dateList:
        # create a filtered dataframe where the last_changed_date column matches the date currently being processed
        datesDataFrame = devicesData.loc[devicesData['lastChangedDate'] == date.date()]
        # the filtered dataframe will retain the indexes, so let's reset them
        datesDataFrame.reset_index(inplace=True)
        # set the "counter variable" as a 0
        datetimeHome = pand.Timedelta(hours=0)

        for item in datesDataFrame.index:
            # if we are working with the first row of the current day, we need to consider the home value differently
            # than the other home values
            if item == 0:
                # If the device wasn't detected at all, save it as such.
                if datesDataFrame.ix[item]['device'] == 0:
                    datetimeHome += (pand.to_datetime(datesDataFrame.ix[item]['lastChanged'])
                                     - pand.to_datetime(pand.to_datetime(datesDataFrame.ix[item]['lastChanged']).date())) * 1

                # If there was multiple instances of the device being detected, collect all of them
                if len(datesDataFrame.index) > 1:
                    datetimeHome += (pand.to_datetime(datesDataFrame.ix[item]['futureChange'])
                                     - pand.to_datetime(datesDataFrame.ix[item]['lastChanged']))\
                                    *datesDataFrame.ix[item]['device']

                # if the device was only seen once, save just that one datapoint
                else:
                    datetimeHome += (pand.to_datetime(datesDataFrame.ix[item]['futureChange'])
                                     - pand.to_datetime(datesDataFrame.ix[item]['lastChanged']))\
                                       * datesDataFrame.ix[item]['device']

            # if this is the last row of this day, calculate the home value until beginning of next day
            elif item == len(datesDataFrame.index) - 1:
                datetimeHome += (pand.to_datetime(((pand.to_datetime(datesDataFrame.ix[item]['lastChanged']))
                                                   + (timedelta(days=1))).date())
                                 - pand.to_datetime(datesDataFrame.ix[item]['lastChanged'])) \
                                   * datesDataFrame.ix[item]['device']

            # otherwise calculate the home value as usual
            else:
                datetimeHome += (pand.to_datetime(datesDataFrame.ix[item]['futureChange'])
                                 - pand.to_datetime(datesDataFrame.ix[item]['lastChanged'])) \
                                * datesDataFrame.ix[item]['device']

    # convert index of our dates dataframe to datetime
    dates.index = pand.to_datetime(dates.index, box=True)

    # convert each value in our dates dataframe from an hour timedelta to an integer
    dates = dates.apply(lambda x: x / nup.timedelta64(1, 'h'))

    # create a series from our dates dataframe (necessary for calmap)
    dataframeSeries = dates['HomeTime']

    # create a calendar map
    calendarPlot = calmap.calendarplot(dataframeSeries, cmap='YlGn', fig_kws=dict(figsize=(15, 3)),
                                     linecolor='white')

    # set the plot title to match our current entity name
    calendarPlot[0].suptitle('Heatmap of number of times wireless device has communicated with{}'.format(entity[0]))
    plt.show()



