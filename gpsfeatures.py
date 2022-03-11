from sqlite3 import Timestamp
from numpy import arccos
import numpy 
import math
import statistics
import pandas as pd  
import csv

from pyparsing import And

#path = "C:\Users\\reeli\OneDrive\Desktop\MQP\dataset\dataset\sensing\gps\\"

class GPS:

    # tmp = dataframe['latitude'].values[n]
    #         tmp2 = dataframe['latitude'].values[n]
    def locationVariance(dataframe):

        #arr = []
        #creates an array that is six elements long in order to calculate the variance by hour
        #for n in range(0, len(dataframe)-1): 
        tmp = numpy.asarray(dataframe['latitude'])
        tmp2 = numpy.asarray(dataframe['longitude'])
        #print("tmp: \n", tmp)
        #print("tmp2: \n", tmp)
        """
            print("tmp from " + str(n) + " to " + str(n+6) + " : ", tmp)
            print("tmp2 from " + str(n) + " to " + str(n+6) + " : ", tmp2)
            print("var from " + str(n) + " to " + str(n+6) + " : ",  statistics.variance(abs(tmp)))
            print("var from " + str(n) + " to " + str(n+6) + " : ",  statistics.variance(abs(tmp2)))
        """
        varSum = statistics.variance(tmp) + statistics.variance(tmp2)
        print("varsum: ", varSum)
        lVarience = math.log(varSum if varSum>0 else 1.0e-100)
        #arr.append(lVarience)
        #arr=numpy.array(arr)
        #print(arr.reshape(-1, 1))
        #return an array that is one column long
        print("lVarience: ", lVarience)
        return lVarience

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
    def speedMean(dataframe):

        userSpeed = 0
        sumofSpeed = 0
        #arr = []
        for n in range(1, len(dataframe)): 
            userlong = dataframe['longitude'].values[n-1]
            userlong2 = dataframe['longitude'].values[n]
            userlat=dataframe['latitude'].values[n-1]
            userlat2=dataframe['latitude'].values[n]
            usertime = dataframe['time'].values[n-1]
            usertime2 = dataframe['time'].values[n]
            userSpeed = numpy.square((userlat2 - userlat)/(usertime2-usertime)) + numpy.square((userlong2 - userlong)/(usertime2-usertime))
            sumofSpeed += numpy.sqrt(userSpeed)    
            meanofSpeed =  (1/len(dataframe))*sumofSpeed 
            #arr.append(meanofSpeed)
            
            #arr=numpy.array(arr)        print("Speed Mean: ", meanofSpeed)

        return meanofSpeed
    

    def totalDistance(dataframe):
        avgD = 0
        arr = []
        #creates an array that is six elements long in order to calculate the distance by hour
        for n in range(1, len(dataframe)): 
            userlong = dataframe['longitude'].values[n-1]
            userlong2 = dataframe['longitude'].values[n]
            userlat=dataframe['latitude'].values[n-1]
            userlat2=dataframe['latitude'].values[n]
            userSum = numpy.square((userlat2 - userlat)) + numpy.square((userlong2 - userlong))
            d = numpy.sqrt(userSum)
            avgD += d
    
        #arr.append(avgD)
        #arr = numpy.array(arr)
        #print("total distance: ", avgD)
        return avgD
    
    #trdef kClusters():

    def transitionTime(dataframe):
        isMoving = 0
        isMoving2 = 0
        time = numpy.asarray(dataframe['time'])
        timeSum = 0
        totalTime = time[-1] - time[0]
        arr = []
        #creates an array that is six elements long in order to calculate the transition time by hour
        for i in range(0, len(dataframe['travelstate'])-1):
            
            if (dataframe['travelstate'].values[i].__eq__("moving")):
                isMoving = time[i]
                isMoving2 = time[i+1]
                timeSum += isMoving2 - isMoving

        transitionRate = timeSum/totalTime
        #arr.append(transitionRate)
        #arr = numpy.array(arr)
        print("transition rate: ", transitionRate)
        return transitionRate
