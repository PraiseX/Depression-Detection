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


import xgboost as xgb                         
  

#au = pd.read_csv('/Users/jessieeteng/Downloads/300_P/300_COVAREP.csv ', sep='\t')



#au = pd.read_csv('/Users/gggwen/Downloads/300_P/300_CLNF_AUs.txt')     

# 1. Note that the action file contains more than features, only use the features when you train your model or normalizers.
# 2. could try use the PHQ8 score directly instead of binary, then after you get predictions you could try different bins to calculate the metrics easily.
# 3.            use the paper we discussed last time to help you recall.
#                 The pipeline should be:
#                                 Iterate through your training user, load all the features and PHQ8 score and concatenate them together
#                                 normalize the corresponding FEATURES
#                                 Train a classifier with training data (features and labels), and save it
#                                 Iterate through all the test user, load the corresponding features and PHQ8 score(ground truth) and concatnate them together
#                                 normalize the features using the trained normalizer
#                                 Make a prediction using the trained classifier and save the prediction
#                                 calculate the metrics (accuracy, f1-score...) based on the prediction and ground truth


#replace with appropriate path
path = "D:\MQP\dataset\DAICWOZ\Facial Features\Action Units\\"
filenames = glob.glob(path + "\*.csv")

#replace with appropriate path
train_list = pd.read_csv('D:\MQP\dataset\DAICWOZ\\train_split_Depression_AVEC2017.csv')                                                         
train_user_id = train_list['Participant_ID']                           
train_PHQ8B = train_list['PHQ8_Score'] 

print("type: ", type(train_PHQ8B))
print("train_PHQ: ", train_PHQ8B)

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est2 = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')


test_list = pd.read_csv('D:\MQP\dataset\DAICWOZ\dev_split_Depression_AVEC2017.csv')                                                         
test_user_id = test_list['Participant_ID']                           
test_PHQ8B = test_list['PHQ8_Score'] 

train_PHQ8B_float = pd.to_numeric(train_PHQ8B,downcast='float')

print("type train_PHQ8B_float: ", type(train_PHQ8B_float))


au_file = [] 
au_file_test = []

#Load train and test features and labels
#make sure all the participants from the training split are in folder
for train_u in train_user_id: 
    #depending on the file use different seperators
    au_file_tmp = pd.read_csv(path+'{}_CLNF_AUs.txt'.format(train_u), sep = ', ') 
    #print(type(au_file_tmp))
    au_file_tmp_mean = au_file_tmp.mean(axis=0)
    au_file_tmp_std = au_file_tmp.std(axis=0) # calculate the std of each column
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    concate  = pd.concat([au_file_tmp_mean, au_file_tmp_std], axis=1)
    au_file.append(concate)
    au_file.append(au_file_tmp[['confidence', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']])  
                                                               
    
for test_u in test_user_id: 
    #depending on the file use different seperators
    au_file_test_tmp = pd.read_csv(path+'{}_CLNF_AUs.txt'.format(train_u), sep = ', ') 
    au_file_test_tmp_mean = au_file_test_tmp.mean(axis=0)
    au_file_test_tmp_std = au_file_test_tmp.std(axis=0) # calculate the std of each column
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    concate  = pd.concat([au_file_test_tmp_mean, au_file_test_tmp_std], axis=1)
    au_file_test.append(concate)
    au_file_test.append(au_file_test_tmp[['confidence', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']]) 
    
    # print("train_u:" , train_u) 
    # print("au_file:" , au_file)                                                             

# avg_score = train_PHQ8B.mean / train_PHQ8B.__len__ - 1                      

# print("mean:", au_file_tmp_mean)
# print("std:", au_file_tmp_std)

au_file = pd.concat(au_file, axis=0)    
au_file_test = pd.concat(au_file_test, axis=0)    



scaler = MinMaxScaler()  


#Normalize train features and apply on test
au_file_transformed = scaler.fit_transform(au_file)  

print("type au file transformed:", au_file_transformed)

au_file_test_transformed = scaler.transform(au_file_test) 

#Initialize classifier
regr = RandomForestClassifier(max_depth=2, random_state=0)              
clf=svm.SVC()
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)


#include naive bayes


#Train the classifier on normalized train features and train labels                 
trainregr3 = regr.fit(au_file_transformed, train_PHQ8B_float) 
#trainregr2 = regr.fit(au_file, train_PHQ8B_2T) 
trainclf3 = clf.fit(au_file_transformed, train_PHQ8B)
#trainclf2 = clf.fit(au_file, train_PHQ8B_2T)
trainxgb3 = xgb_model.fit(au_file_transformed, train_PHQ8B)
#trainxgb2 = xgb_model.fit(au_file, train_PHQ8B_2T)
#include naive bayes


#Make predictions on normalized test features, which will gives you the predicted labels
predictionRegr = regr.predict(au_file_test_transformed)   
predictionSVM = clf.predict(au_file_test_transformed)                             
predictionXGB = xgb_model.predict(au_file_test_transformed)

#Divide the test labels and predicted labels into K-bins and calculate corresponding metrics.
train_PHQ8B_3BT = est.fit_transform(train_PHQ8B)
test_PHQ8B_3BT = est.transform(test_PHQ8B)
predictionRegr_3BT = est.transform(predictionRegr)
predictionSVM_3BT = est.transform(predictionSVM)
predictionXGB_3BT = est.transform(predictionXGB)


train_PHQ8B_2BT = est2.fit_transform(train_PHQ8B)
test_PHQ8B_2BT = est2.transform(test_PHQ8B)
predictionRegr_2BT = est2.transform(predictionRegr)
predictionSVM_2BT = est2.transform(predictionSVM)
predictionXGB_2BT = est2.transform(predictionXGB)

predictprobRFC = regr.predict_proba(au_file_test)
predictprobSVM = clf.predict_proba(au_file_test)
predictprobXGB = xgb_model.classes_

twobinrfc = len(predictprobRFC) / 2.0
twobinsvm = len(predictprobSVM) / 2.0

rfc = np.add.reduceat(predictprobRFC, np.arange(0, len(predictprobRFC), twobinrfc))


print("predict_prob: ", predictprobRFC)


#metrics for regression
#regrROC3regr = sklearn.metrics.roc_auc_score(test_PHQ8B_3BT, clf.predict_proba(au_file_test))
regrROC2regr = sklearn.metrics.roc_auc_score(test_PHQ8B_2BT, clf.predict_proba(au_file_test))
accuracy3regr = accuracy_score(test_PHQ8B_3BT, predictionRegr_3BT)
accuracy2regr = accuracy_score(test_PHQ8B_2BT, predictionRegr_2BT)
f13regr = f1_score(test_PHQ8B_3BT, predictionRegr_3BT)
f12regr = f1_score(test_PHQ8B_2BT, predictionRegr_2BT)
tn3regr, fp3regr, fn3regr, tp3regr = confusion_matrix(test_PHQ8B_3BT, predictionRegr_3BT).ravel()
tn2regr, fp2regr, fn2regr, tp2regr = confusion_matrix(test_PHQ8B_2BT, predictionRegr_2BT).ravel()


# probability = clf.predict_proba(au_file_test)
# probability

#metrics for svm
#regrROC3svm = sklearn.metrics.roc_auc_score(test_PHQ8B, clf.predict_proba(au_file_test))
#returns array
#divide 
regrROC2svm = sklearn.metrics.roc_auc_score(test_PHQ8B_2BT, clf.predict_proba(au_file_test))
accuracy3svm = accuracy_score(test_PHQ8B_3BT, predictionRegr_3BT)
accuracy2svm = accuracy_score(test_PHQ8B_2BT, predictionRegr_2BT)
f13svm = f1_score(test_PHQ8B_3BT, predictionRegr_3BT)
f12svm = f1_score(test_PHQ8B_2BT, predictionRegr_2BT)
tn3svm, fp3svm, fn3svm, tp3svm = confusion_matrix(test_PHQ8B_3BT, predictionRegr_3BT).ravel()
tn2svm, fp2svm, fn2svm, tp2svm = confusion_matrix(test_PHQ8B_2BT, predictionRegr_2BT).ravel()

#metrics for xgb
#regrROC3xgb = sklearn.metrics.roc_auc_score(test_PHQ8B_3BT, clf.predict_proba(au_file_test))
# regrROC2xgb = sklearn.metrics.roc_auc_score(test_PHQ8B_2BT, clf.predict_proba(au_file_test))
accuracy3xgb = accuracy_score(test_PHQ8B_3BT, predictionRegr_3BT)
accuracy2xgb = accuracy_score(test_PHQ8B_2BT, predictionRegr_2BT)
f13xgb = f1_score(test_PHQ8B_3BT, predictionRegr_3BT)
f12xgb = f1_score(test_PHQ8B_2BT, predictionRegr_2BT)
tn3xgb, fp3xgb, fn3xgb, tp3xgb = confusion_matrix(test_PHQ8B_3BT, predictionRegr_3BT).ravel()
tn2xgb, fp2xgb, fn2xgb, tp2xgb = confusion_matrix(test_PHQ8B_2BT, predictionRegr_2BT).ravel()


print("fiscore:", f12svm)




                                                                                
 