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
        arr = []

        #creates an array that is six elements long in order to calculate the variance by hour
        for n in range(0, len(latitude)-1, 7):
            for m in range(0, len(longitude)-1, 7):
                tmp = latitude[n:n+6]
                tmp2 = longitude[m:m+6]
                """
                print("tmp from " + str(n) + " to " + str(n+6) + " : ", tmp)
                print("tmp2 from " + str(n) + " to " + str(n+6) + " : ", tmp2)
                print("var from " + str(n) + " to " + str(n+6) + " : ",  statistics.variance(abs(tmp)))
                print("var from " + str(n) + " to " + str(n+6) + " : ",  statistics.variance(abs(tmp2)))
                """
                varSum = statistics.variance(tmp) + statistics.variance(abs(tmp2))
                lVarience = math.log(varSum if varSum > 0 else 1)
                arr.append(lVarience)

        arr=numpy.array(arr)

        print(arr.reshape(1,-1))

        #return an array that is one column long
        return arr.reshape(1,-1)

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
            sumofSpeed = 0
            arr = []
            for n in range(0, len(timeStamps)-1, 7):

                #gets the spped mean hourly
                while(n+6<len(timeStamps)-1):
                    userSpeed = numpy.square((float(latitude[n+6]) - float(latitude[n]))/(float(timeStamps[n+6])-float(timeStamps[n]))) + numpy.square((float(longitude[n+6]) - float(longitude[n]))/(float(timeStamps[n+6])-float(timeStamps[n])))
                    sumofSpeed = numpy.sqrt(userSpeed)
                    meanofSpeed =  (1/len(timeStamps)-1)*sumofSpeed 
                    arr.append(meanofSpeed)
            arr=numpy.array(arr)
            return arr.reshape(1,-1)
    

    def totalDistance(latitude, longitude, timeStamps):
        avgD = 0
        arr = []
        #creates an array that is six elements long in order to calculate the distance by hour
        for n in range(0, len(latitude)-1, 7):
            for m in range(0, len(longitude)-1, 7):
                tmp = latitude[n:n+6]
                tmp2 = longitude[n:n+6]
                d = numpy.square(latitude[n+1]-latitude[n]) + numpy.square(longitude[m+1]+longitude[m])
                avgD = numpy.sqrt(d)
                avgD+=avgD
                arr.append(avgD)
        arr = numpy.array(arr)
        return arr.reshape(1,-1)
    
    #trdef kClusters():

    def transitionTime(travelState):
        timeMoving = 0
        arr = []
        #creates an array that is six elements long in order to calculate the transition time by hour
        for i in range(0, len(travelState)-1, 7):
            tmp = travelState[i:i+6]
            for i in range(0, len(tmp)-1):
                if tmp[i].__eq__("moving"):
                    timeMoving+=1
            transitionRate = timeMoving/len(travelState)-1
            arr.append(transitionRate)
        arr = numpy.array(arr)
        return arr.reshape(1,-1)
