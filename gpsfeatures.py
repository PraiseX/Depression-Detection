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
            for n in range(1, len(timeStamps)):
                print("timestamp:", timeStamps)
                print("timeStamp",  timeStamps[n])
                print("timestamp n+1:", timeStamps[n+1])
                print("latitude n+1 type:", latitude[n+1])
                print("latitude n type:", latitude[n])
                print("latitude n+1:", longitude[n+1])
                print("latitude n:", longitude[n])

                userSpeed = numpy.square((float(latitude[n+1]) - float(latitude[n+1]))/(float(timeStamps[n+1])-float(timeStamps[n+1]))) + numpy.square((float(longitude[n]) - float(longitude[n]))/(float(timeStamps[n])-float(timeStamps[n])))
                meanofSpeed = numpy.sqrt(userSpeed)
                meanofSpeed+=meanofSpeed
            return (1/len(timeStamps))*meanofSpeed
    
    def totalDistance(latitude1, longitude1, latitude2, longitude2, timeStamps):
        for i in range(1, timeStamps):
            d = numpy.square(latitude2-latitude1) + numpy.square(longitude2+longitude1)
            avgD = numpy.sqrt(d)
            avgD+=avgD
    
    #trdef kClusters():

    def transitionTime(travelState):
        timeMoving = 0
        for i in travelState:
            if travelState.__eq__("moving"):
                timeMoving+=1
        return timeMoving/len(travelState)
