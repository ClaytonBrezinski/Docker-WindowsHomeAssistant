from __future__ import division

import numpy as nump
import pandas as pand
import collections

from itertools import izip, count
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
from sklearn.cluster import KMeans

from random import randint
from matplotlib import style

from twilio.rest import Client


def getMovingAverage(inputData, windowSize):
    """
    gets the moving average of a system using linear convolution
    :param inputData: pandas series dataset of the data collected from the database
    :param windowSize: integer of how long of a timeperiod we should consider the moving average
    :return: a numpy array of the linear convolution that took place.
    """
    window = nump.ones(int(windowSize)) / float(windowSize)
    return nump.convolve(inputData, window, 'same')


def identifyIrregularity(inputData, windowSize, standardDeviation=1.0):
    """
    identifies irregularities within the dataset
    :param inputData: pandas series dataset of the data collected from the database
    :param windowSize: integer of how long of a timeperiod we should consider the moving average
    :param standardDeviation: integer representing standard deviation
    :return: dictionary of: standardDeviation, anomalies
    """

    average = getMovingAverage(inputData, windowSize).tolist()

    # Calculate the variation in the distribution
    standard = nump.std(average)
    standard = cluster(standard)
    return {'standardDeviation': round(standard, 3),
            'anomaliesDict': collections.OrderedDict(
                    [(index, inputDataI) for index, inputDataI, AverageI in izip(count(), inputData, average)
                     if (inputDataI > AverageI + (standardDeviation * standard)) | (inputDataI < AverageI - (standardDeviation * standard))]
                    )}


def getRollingStandardDeviation(inputData, windowSize, standardDeviation=1.0):
    """
    another method of identifying irregularities within the dataset
    :param inputData: pandas series dataset of the data collected from the database
    :param windowSize: integer of how long of a timeperiod we should consider the moving average
    :param standardDeviation: integer representing standard deviation
    :return: dictionary of: standardDeviation, anomalies
    """
    average = getMovingAverage(inputData, windowSize)
    listOfAverages = average.tolist()
    residual = inputData - average
    # Calculate the variation in the distribution of the residual
    variationDataframe = pand.DataFrame(pand.rolling_std(residual, windowSize))
    rollingStandardDeviation = variationDataframe.replace(nump.nan, variationDataframe.ix[windowSize - 1]).tolist()
    std = nump.std(residual)
    return {'standardDeviation': round(std, 3),
            'anomaliesDict': collections.OrderedDict(
                    [(index, inputDataI)
                     for index, inputDataI, AverageI, RSI in izip(count(), inputData, listOfAverages, rollingStandardDeviation)
                     if (inputDataI > AverageI + (standardDeviation * RSI)) | (inputDataI < AverageI - (standardDeviation * RSI))])}


def cluster(inputData, kMeans=range(1, 20)):
    """
    Use K means clustering in order to find common patterns within the routes taken within the house.
    :param inputData: pandas series dataset of the data collected from the database on route taken within the house
    :param kMeans: the value K that we want to use as the clustering variable
    :return: list of the
    """
    numSamples, numFeatures = inputData.shape
    numDataPoints = len(nump.unique(inputData.target))
    labels = inputData.target
    varianceClusters = []
    for k in kMeans:
        reducedData = PCA(n_components=2).fit_transform(inputData)
        kMeans = KMeans(init='k-means++', n_clusters=k, n_init=k)
        kMeans.fit(reducedData)
        varianceClusters.append(sum(nump.min(cdist(reducedData, kMeans.cluster_centers_, 'euclidean'),
                                             axis=1)) / inputData.shape[0])
    return varianceClusters


# This function is repsonsible for displaying how the function performs on the given dataset.
def determineIrregulaityOccurence(inputData, windowSize, standardDeviation=1, rollingStandard=False):
    # Query for the anomalies and in the system and if they are found, send text to Admin user



def notify(locationName):
    # provide ID and token to gain access to REST API
    client = Client("X", "X")

    # change the "from_" number to your Twilio number and the "to" number
    # to the phone number you signed up for Twilio with, or upgrade your
    # account to send SMS to any phone number
    client.messages.create(to="+",
                           from_="+",
                           body="Home assistant machine learning detected an anomaly at %s" % locationName)

if __name__ == "__main__":
    DB_URL = "sqlite:///./home-assistant_v2.db"
    engine = create_engine(DB_URL)

    # gather entities from the database
    entities = engine.execute(
            """SELECT entity_id, COUNT(*) FROM states WHERE commsDevice = "Clayton's phone"  GROUP BY entity_id""")

    dataFrame = pand.DataFrame(entities, columns=['Months', 'SunSpots'])
    dataFrame.head()

    rollingStandard = getRollingStandardDeviation(dataFrame, 10, 1.0)

    if rollingStandard:
        events = getRollingStandardDeviation(dataFrame, 10, 1.0)
    else:
        events = identifyIrregularity(dataFrame, 10, 1.0)

    if nump.fromiter(events['anomalies_dict'].itervalues(), dtype=float, count=len(events['anomalies_dict'])) > 1 \
            | nump.fromiter(events['anomalies_dict'].iterkeys(), dtype=int, count=len(events['anomalies_dict'])) > 1:
        notify(str(nump.fromiter(events['anomalies_dict'].deviceId))