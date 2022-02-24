from sqlite3 import Timestamp
import sklearn
import os
import pandas as pd  
import glob
import numpy as np  
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
print("PHQUSERID:", np.asarray(phquserID))
phquserID=np.asarray(phquserID)


for i, idx in enumerate(phquserID):
    gpsFile=pd.read_csv(path+'gps_{}.csv'.format(phquserID[i]), index_col=False)
    #print("file:", gpsFile)
    userLatitude = gpsFile[['latitude']]
    userLatitude = np.asarray(userLatitude)
    #print("userLatitude: ", userLatitude)
    userLongitude = gpsFile[['longitude']]
    userLongitude = np.asarray(userLongitude)
    timestamps=gpsFile[['time']]
    timestamps = np.asarray(timestamps)
    travelState = gpsFile[['travelstate']]
    #print("travelstate: ", np.asarray(travelState))
    print("locationVarience for user " + phquserID[i] + ":", GPS.locationVariance(userLatitude.flatten('A'), userLongitude.flatten('A')))
    print("SpeedMean for user " + phquserID[i] + ":", GPS.speedMean(userLatitude, userLongitude, timestamps))
    print("Total Distance for user " + phquserID[i] + ":",GPS.totalDistance(userLatitude, userLongitude, timestamps))
    print("Transistion Time for user " + phquserID[i] + ":",GPS.transitionTime(timestamps))