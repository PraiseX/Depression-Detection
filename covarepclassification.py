import sklearn
import os
import pandas as pd  
import glob
import numpy as np  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import multiclass
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb                         
import joblib  

#replace with appropriate path
path = "D:\MQP\dataset\DAICWOZ\Audio Features\\COVAREP\\"
filenames = glob.glob(path + "\*.csv")

#replace with appropriate path
train_list = pd.read_csv('D:\MQP\dataset\DAICWOZ\\train_split_Depression_AVEC2017.csv')                                                         
train_user_id = train_list['Participant_ID']                           
train_PHQ8B = train_list[['PHQ8_Score']] 
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


test_list = pd.read_csv('D:\MQP\dataset\DAICWOZ\dev_split_Depression_AVEC2017.csv')                                                         
test_user_id = test_list['Participant_ID']                           
test_PHQ8B = test_list[['PHQ8_Score']] 
groundtruth = est.transform(np.array(test_PHQ8B)).flatten()
groundtruth2 = est2.transform(np.array(test_PHQ8B)).flatten()

gaze_file = [] 
gaze_file_test = []

PHQ_score = []
PHQ_score_test = []

gaze_indexes = [0]
gaze_file_len_idx = 0

print("---loading Gaze Files----")

for idx, train_u in enumerate(train_user_id):
    #depending on the file use different seperators
    gaze_file_tmp = pd.read_csv(path+'{}_COVAREP.csv'.format(train_u), sep = ', ')
    #au_file_tmp_mean = au_file_tmp.iloc[:, -22:].mean(axis=0)
    #au_file_tmp_std = au_file_tmp.iloc[:, -22:].std(axis=0)
    #print(type(au_file_tmp))
    gaze_file_len = len(gaze_file_tmp)
    gaze_score = train_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_tmp = [gaze_score]*gaze_file_len
    #print("AU score:", au_score)
    PHQ_score.extend(PHQ_score_tmp)

    #print("is au tmp any na:", pd.isna(au_file_tmp).any())
     # calculate the std of each column
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    #concate  = pd.concat([au_file_tmp_mean, au_file_tmp_std], axis=0)
    #au_file.append(concate)
    gaze_file.append(gaze_file_tmp[['confidence', 'success', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_h0', 'y_h0', 'z_h0', 'x_h1', 'y_h1', 'z_h1']])


for idx, test_u in enumerate(test_user_id): 
    #depending on the file use different seperators
    gaze_file_test_tmp = pd.read_csv(path+'{}_COVAREP.csv'.format(test_u), sep = ', ')
    gaze_file_test_tmp_mean = gaze_file_test_tmp.iloc[:, -206:].mean(axis=0)
    gaze_file_test_tmp_std = gaze_file_test_tmp.iloc[:, -206:].std(axis=0)
    #print("is au tmp test any na:", pd.isna(au_file_test_tmp).any())
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    gaze_file_len = len(gaze_file_test_tmp)
    gaze_file_len_idx += gaze_file_len
    gaze_indexes.append(gaze_file_len_idx)

    gaze_score = test_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_test_tmp = [gaze_score]*gaze_file_len
    PHQ_score_test.extend(PHQ_score_test_tmp)
    concate  = pd.concat([gaze_file_test_tmp_mean, gaze_file_test_tmp_std], axis=1)
    #au_file_test.append(concate)
    gaze_file_test.append(gaze_file_test_tmp[['confidence', 'success', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_h0', 'y_h0', 'z_h0', 'x_h1', 'y_h1', 'z_h1']])

    
