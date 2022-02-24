from sqlite3 import Timestamp
from numpy import arccos
import numpy
import math
import statistics
import pandas as pd  
import csv

#path = "C:\Users\\reeli\OneDrive\Desktop\MQP\dataset\dataset\sensing\gps\\"

class GPS:

   
    def locationVariance(latitude, longitude):
          return math.log(statistics.variance(latitude)+ statistics.variance(longitude))

    """
    def getTimeStamps(filename):
        timeStampNum = 0
        for idx, train_u in enumerate(filename):
            gps_file = pd.read_csv(path+'gps_u{}.csv'.format(train_u))
            timecolumn = gps_file.iloc[0]
            if timecolumn.any(0):
                timeStampNum+=1
        return timeStampNum
    """
    def speedMean(latitude, longitude, timeStamps):
            userSpeed = 0
            meanofSpeed = 0
            for n in range(0, len(timeStamps)-1):
                userSpeed = numpy.square((float(latitude[n+1]) - float(latitude[n]))/(float(timeStamps[n+1])-float(timeStamps[n]))) + numpy.square((float(longitude[n+1]) - float(longitude[n]))/(float(timeStamps[n+1])-float(timeStamps[n])))
                meanofSpeed = numpy.sqrt(userSpeed)
                meanofSpeed+=meanofSpeed
            return (1/len(timeStamps)-1)*meanofSpeed
    
    def totalDistance(latitude, longitude, timeStamps):
        avgD = 0
        for i in range(0, len(timeStamps)-1):
            d = numpy.square(latitude[i+1]-latitude[i]) + numpy.square(longitude[i+1]+longitude[i])
            avgD = numpy.sqrt(d)
            avgD+=avgD
        return avgD
    
    #trdef kClusters():

    def transitionTime(travelState):
        timeMoving = 0
        for i in range(0, len(travelState)-1):
            if travelState[i].__eq__("moving"):
                timeMoving+=1
        return timeMoving/len(travelState)-1
