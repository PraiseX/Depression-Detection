import sklearn
import os
import pandas as pd  
import glob
import numpy as np  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import multiclass
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb                         
import joblib  

#replace with appropriate path
path = "C:\Users\\reeli\OneDrive\Desktop\MQP\Depression-Detection\gpsFeatures\\"
filenames = glob.glob(path + "\*.csv")

#replace with appropriate path

train_list = pd.read_csv('C:\Users\\reeli\OneDrive\Desktop\MQP\Depression-Detection\postScoresTrain.csv')                                                         
train_user_id = train_list['uid']                           
train_PHQ8B = train_list[['Score']] 
train_PHQ8B = np.reshape(train_PHQ8B, (-1,1))

#print("Train PHQ8 reshaped:", train_PHQ8B)
#TrainPHQ_score_array = np.array(train_PHQ8B)



# print("type: ", type(train_PHQ8B))
#print("train_PHQ: ", train_PHQ8B.iloc[1, :])

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est2 = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')


#train_PHQ8BT = np.reshape(train_PHQ8B,(-1,1))
train_PHQ8B_3BT = est.fit_transform(train_PHQ8B)
#print("3 bin edges:", est.bin_edges_)
train_PHQ8B_2BT = est2.fit_transform(train_PHQ8B)
#print("2 bin edges:", est2.bin_edges_)


test_list = pd.read_csv('C:\Users\\reeli\OneDrive\Desktop\MQP\Depression-Detection\postScoresTest.csv')                                                         
test_user_id = test_list['uid']                           
test_PHQ8B = test_list[['Score']] 
groundtruth = est.transform(np.array(test_PHQ8B)).flatten()
groundtruth2 = est2.transform(np.array(test_PHQ8B)).flatten()

gps_file = [] 
gps_file_test = []

PHQ_score = []
PHQ_score_test = []

gps_indexes = [0]
gps_file_len_idx = 0

print("---loading gps Files----")

for idx, train_u in enumerate(train_user_id):
    #depending on the file use different seperators
    gps_file_tmp = pd.read_csv(path+'gpsFeature{}.csv'.format(train_u), sep = ',')
    #au_file_tmp_mean = au_file_tmp.iloc[:, -22:].mean(axis=0)
    #au_file_tmp_std = au_file_tmp.iloc[:, -22:].std(axis=0)
    #print(type(au_file_tmp))
    gps_file_len = len(gps_file_tmp)
    gps_score = train_PHQ8B['Score'].iloc[idx]
    PHQ_score_tmp = [gps_score]*gps_file_len
    #print("AU score:", au_score)
    PHQ_score.extend(PHQ_score_tmp)

    #print("is au tmp any na:", pd.isna(au_file_tmp).any())
     # calculate the std of each column
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    #concate  = pd.concat([au_file_tmp_mean, au_file_tmp_std], axis=0)
    #au_file.append(concate)
    gps_file.append(gps_file_tmp[['location varience','speed mean','total distance','transition time']])



for idx, test_u in enumerate(test_user_id): 
    #depending on the file use different seperators
    gps_file_test_tmp = pd.read_csv(path+'gpsFeature{}.csv'.format(test_u), sep = ',')
    #gps_file_test_tmp_mean = gps_file_test_tmp.iloc[:, -8:].mean(axis=0)
    #gps_file_test_tmp_std = gps_file_test_tmp.iloc[:, -8:].std(axis=0)
    #print("is au tmp test any na:", pd.isna(au_file_test_tmp).any())
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    gps_file_len = len(gps_file_test_tmp)
    gps_file_len_idx += gps_file_len
    gps_indexes.append(gps_file_len_idx)

    gps_score = test_PHQ8B['Score'].iloc[idx]
    PHQ_score_test_tmp = [gps_score]*gps_file_len
    PHQ_score_test.extend(PHQ_score_test_tmp)
    #concate  = pd.concat([gps_file_test_tmp_mean, gps_file_test_tmp_std], axis=1)
    #au_file_test.append(concate)
    gps_file_test.append(gps_file_test_tmp[['location varience','speed mean','total distance','transition time']])
    # print("train_u:" , train_u) 
    # print("au_file:" , au_file)                                                             

print("---done loading gps Files----")
PHQ_score_array = np.array(PHQ_score)
PHQ_score_R = np.reshape(PHQ_score_array,(-1,1))
PHQ_score_test_array = np.array(PHQ_score_test)
PHQ_score_test_R = np.reshape(PHQ_score_test_array,(-1,1))
gps_file = pd.concat(gps_file, axis=0)    
gps_file_test = pd.concat(gps_file_test, axis=0)    
gps_indexes = np.array(gps_indexes)
gps_indexes_for_sum = gps_indexes[1:]-gps_indexes[:-1]
gps_indexes_for_sum = np.reshape(gps_indexes_for_sum, (-1,1))

#print("2D indexes for sum:", gps_indexes_for_sum)
#print("2D indexes for sum shape:", gps_indexes_for_sum.shape)
#print("2D indexes for sum shape -5:", gps_indexes_for_sum[:-5].shape)


#test = np.array(test_PHQ8B)
#print("is au any na:", pd.isna(au_file).any())
#print("is au test any na:", pd.isna(au_file_test).any())
scaler = MinMaxScaler()  
#Normalize train features and apply on test
gps_file_transformed = scaler.fit_transform(gps_file)  
gps_file_test_transformed = scaler.transform(gps_file_test) 
