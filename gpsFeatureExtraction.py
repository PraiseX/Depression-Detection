from sqlite3 import Timestamp
import sklearn
import os
import pandas as pd  
import glob
import numpy as np  
import pickle
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import multiclass
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb                         
import joblib  
import math
import statistics

from gpsfeatures import GPS

path = "C:/Users/reeli/OneDrive/Desktop/MQP/dataset/dataset/sensing/gps/"
phqList=pd.read_csv(r"C:\Users\reeli\OneDrive\Desktop\MQP\dataset\dataset\survey\PHQ-9.csv")
phquserID=phqList['uid']
#print("PHQUSERID:", np.asarray(phquserID))
phquserID=np.asarray(phquserID)
arr = []

for i, idx in enumerate(phquserID):
    gpsFile=pd.read_csv(path+'gps_{}.csv'.format(phquserID[i]), index_col=False)
    #print("gps file columns: \n", gpsFile.columns.tolist)
    #gpsSeries = pd.Series(gpsFile)  
    #print("file:", gpsFile)
    #userLatitude = gpsFile[['latitude']]
    #userLatitude = np.asarray(userLatitude)
    #print("userLatitude: ", userLatitude)
    #userLongitude = gpsFile[['longitude']]
    #userLongitude = np.asarray(userLongitude)
    usertime=pd.DataFrame(gpsFile['time'])
    #print("user " + str(phquserID[i]) + " length", len(usertime))
    #print("usertime for" + str(phquserID[i]) + ": ", usertime)
    #timestamps = np.asarray(timestamps)
    #travelState = gpsFile[['travelstate']]

    location_variance, speed_mean, total_distance, transition_time = [], [], [], [] 

    starttime = usertime.values[0]
    endttime =  usertime.values[-1]
    step = 7200
    #print("startime: ", starttime)
    #print("endtime: ", endttime)
    for periodStart in np.arange(starttime, endttime, step):
        periodEnd = periodStart+step
        tmpdata = gpsFile[(gpsFile['time']>=periodStart) & (gpsFile['time']<periodEnd)] 
        #print("tmpdata latituude for " + str(phquserID[i]) +":\n", len(np.asarray(tmpdata['latitude'])) )
        while not arr:    
            location_variance.append(GPS.locationVariance(tmpdata))
            speed_mean.append(GPS.speedMean(tmpdata))
            total_distance.append(GPS.totalDistance(tmpdata))
            transition_time.append(GPS.transitionTime(tmpdata))

    result = pd.DataFrame()
    result['location varience'] = location_variance
    result['speed mean'] = speed_mean
    result['total distance'] = total_distance
    result['transition time'] = transition_time
    result.to_csv('gpsFeature' + str(phquserID[i]) + '.csv', index=False)


    #starttime = gpsFile[]
   
    
  