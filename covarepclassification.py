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

covarep_file = [] 
covarep_file_test = []

PHQ_score = []
PHQ_score_test = []

covarep_indexes = [0]
covarep_file_len_idx = 0

print("---loading Covarep Files----")

for idx, train_u in enumerate(train_user_id):
    #depending on the file use different seperators
    covarep_file_tmp = pd.read_csv(path+'{}_COVAREP.csv'.format(train_u), names = ['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'peakSlope', 'Rd', 'Rd_conf', 'creak', 'MCEP_0', 'MCEP_1', 'MCEP_2', 'MCEP_3', 'MCEP_4', 'MCEP_5', 'MCEP_6', 'MCEP_7', 'MCEP_8', 'MCEP_9', 'MCEP_10', 'MCEP_11', 'MCEP_12', 'MCEP_13', 'MCEP_14', 'MCEP_15', 'MCEP_16', 'MCEP_17', 'MCEP_18', 'MCEP_19', 'MCEP_20', 'MCEP_21', 'MCEP_22', 'MCEP_23','MCEP_24', 'HMPDM_0', 'HMPDM_1', 'HMPDM_2', 'HMPDM_3', 'HMPDM_4', 'HMPDM_5', 'HMPDM_6', 'HMPDM_7', 'HMPDM_8', 'HMPDM_9', 'HMPDM_10', 'HMPDM_11', 'HMPDM_12', 'HMPDM_13', 'HMPDM_14', 'HMPDM_15', 'HMPDM_16', 'HMPDM_17', 'HMPDM_18', 'HMPDM_19', 'HMPDM_20', 'HMPDM_21', 'HMPDM_22', 'HMPDM_23', 'HMPDM_24', 'HMPDD_0', 'HMPDD_1', 'HMPDD_2', 'HMPDD_3', 'HMPDD_4', 'HMPDD_5', 'HMPDD_6', 'HMPDD_7', 'HMPDD_8', 'HMPDD_9', 'HMPDD_10', 'HMPDD_11', 'HMPDD-12']
 ,sep = ', ')
    #au_file_tmp_mean = au_file_tmp.iloc[:, -22:].mean(axis=0)
    #au_file_tmp_std = au_file_tmp.iloc[:, -22:].std(axis=0)
    #print(type(au_file_tmp))
    covarep_file_len = len(covarep_file_tmp)
    covarep_score = train_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_tmp = [covarep_score]*covarep_file_len
    #print("AU score:", au_score)
    PHQ_score.extend(PHQ_score_tmp)

    #print("is au tmp any na:", pd.isna(au_file_tmp).any())
     # calculate the std of each column
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    #concate  = pd.concat([au_file_tmp_mean, au_file_tmp_std], axis=0)
    #au_file.append(concate)
    covarep_file.append(covarep_file_tmp[['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'peakSlope', 'Rd', 'Rd_conf', 'creak', 'MCEP_0', 'MCEP_1', 'MCEP_2', 'MCEP_3', 'MCEP_4', 'MCEP_5', 'MCEP_6', 'MCEP_7', 'MCEP_8', 'MCEP_9', 'MCEP_10', 'MCEP_11', 'MCEP_12', 'MCEP_13', 'MCEP_14', 'MCEP_15', 'MCEP_16', 'MCEP_17', 'MCEP_18', 'MCEP_19', 'MCEP_20', 'MCEP_21', 'MCEP_22', 'MCEP_23','MCEP_24', 'HMPDM_0', 'HMPDM_1', 'HMPDM_2', 'HMPDM_3', 'HMPDM_4', 'HMPDM_5', 'HMPDM_6', 'HMPDM_7', 'HMPDM_8', 'HMPDM_9', 'HMPDM_10', 'HMPDM_11', 'HMPDM_12', 'HMPDM_13', 'HMPDM_14', 'HMPDM_15', 'HMPDM_16', 'HMPDM_17', 'HMPDM_18', 'HMPDM_19', 'HMPDM_20', 'HMPDM_21', 'HMPDM_22', 'HMPDM_23', 'HMPDM_24', 'HMPDD_0', 'HMPDD_1', 'HMPDD_2', 'HMPDD_3', 'HMPDD_4', 'HMPDD_5', 'HMPDD_6', 'HMPDD_7', 'HMPDD_8', 'HMPDD_9', 'HMPDD_10', 'HMPDD_11', 'HMPDD-12']])
    #F0, VUV, NAQ, QOQ, H1H2, PSP, MDQ, peakSlope, Rd, Rd_conf, MCEP_0-24,HMPDM_0-24, HMPDD_0-12


for idx, test_u in enumerate(test_user_id): 
    #depending on the file use different seperators
    covarep_file_test_tmp = pd.read_csv(path+'{}_COVAREP.csv'.format(test_u), names = ['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'peakSlope', 'Rd', 'Rd_conf', 'creak', 'MCEP_0', 'MCEP_1', 'MCEP_2', 'MCEP_3', 'MCEP_4', 'MCEP_5', 'MCEP_6', 'MCEP_7', 'MCEP_8', 'MCEP_9', 'MCEP_10', 'MCEP_11', 'MCEP_12', 'MCEP_13', 'MCEP_14', 'MCEP_15', 'MCEP_16', 'MCEP_17', 'MCEP_18', 'MCEP_19', 'MCEP_20', 'MCEP_21', 'MCEP_22', 'MCEP_23','MCEP_24', 'HMPDM_0', 'HMPDM_1', 'HMPDM_2', 'HMPDM_3', 'HMPDM_4', 'HMPDM_5', 'HMPDM_6', 'HMPDM_7', 'HMPDM_8', 'HMPDM_9', 'HMPDM_10', 'HMPDM_11', 'HMPDM_12', 'HMPDM_13', 'HMPDM_14', 'HMPDM_15', 'HMPDM_16', 'HMPDM_17', 'HMPDM_18', 'HMPDM_19', 'HMPDM_20', 'HMPDM_21', 'HMPDM_22', 'HMPDM_23', 'HMPDM_24', 'HMPDD_0', 'HMPDD_1', 'HMPDD_2', 'HMPDD_3', 'HMPDD_4', 'HMPDD_5', 'HMPDD_6', 'HMPDD_7', 'HMPDD_8', 'HMPDD_9', 'HMPDD_10', 'HMPDD_11', 'HMPDD-12'], sep = ', ')
    #covarep_file_test_tmp_mean = covarep_file_test_tmp.iloc[:, -206:].mean(axis=0)
    #covarep_file_test_tmp_std = covarep_file_test_tmp.iloc[:, -206:].std(axis=0)
    #print("is au tmp test any na:", pd.isna(au_file_test_tmp).any())
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    covarep_file_len = len(covarep_file_test_tmp)
    covarep_file_len_idx += covarep_file_len
    covarep_indexes.append(covarep_file_len_idx)

    covarep_score = test_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_test_tmp = [covarep_score]*covarep_file_len
    PHQ_score_test.extend(PHQ_score_test_tmp)
    #concate  = pd.concat([covarep_file_test_tmp_mean, covarep_file_test_tmp_std], axis=1)
    #au_file_test.append(concate)
    covarep_file_test.append(covarep_file_test_tmp[['F0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', 'peakSlope', 'Rd', 'Rd_conf', 'creak', 'MCEP_0', 'MCEP_1', 'MCEP_2', 'MCEP_3', 'MCEP_4', 'MCEP_5', 'MCEP_6', 'MCEP_7', 'MCEP_8', 'MCEP_9', 'MCEP_10', 'MCEP_11', 'MCEP_12', 'MCEP_13', 'MCEP_14', 'MCEP_15', 'MCEP_16', 'MCEP_17', 'MCEP_18', 'MCEP_19', 'MCEP_20', 'MCEP_21', 'MCEP_22', 'MCEP_23','MCEP_24', 'HMPDM_0', 'HMPDM_1', 'HMPDM_2', 'HMPDM_3', 'HMPDM_4', 'HMPDM_5', 'HMPDM_6', 'HMPDM_7', 'HMPDM_8', 'HMPDM_9', 'HMPDM_10', 'HMPDM_11', 'HMPDM_12', 'HMPDM_13', 'HMPDM_14', 'HMPDM_15', 'HMPDM_16', 'HMPDM_17', 'HMPDM_18', 'HMPDM_19', 'HMPDM_20', 'HMPDM_21', 'HMPDM_22', 'HMPDM_23', 'HMPDM_24', 'HMPDD_0', 'HMPDD_1', 'HMPDD_2', 'HMPDD_3', 'HMPDD_4', 'HMPDD_5', 'HMPDD_6', 'HMPDD_7', 'HMPDD_8', 'HMPDD_9', 'HMPDD_10', 'HMPDD_11', 'HMPDD-12']])
    
    #One	important	aspect	is	that	VUV (voiced/unvoiced) provides	a	flag	({0,1})	if	the	
    #current	segment	is	voiced	or	unvoiced.	In	unvoiced	case,	i.e.	VUV	=	0,	the	vocal	
    #folds	are	detected	to	not	be	vibrating,	hence	values	such	as	F0,	NAQ,	QOQ,	
    #H1H2,	PSP,	MDQ,	peakSlope,	and	Rd	should	not	be	utilized

print("---done loading Covarep Files----")


PHQ_score_array = np.array(PHQ_score)
PHQ_score_R = np.reshape(PHQ_score_array,(-1,1))
PHQ_score_test_array = np.array(PHQ_score_test)
PHQ_score_test_R = np.reshape(PHQ_score_test_array,(-1,1))
covarep_file = pd.concat(covarep_file, axis=0)    
covarep_file_test = pd.concat(covarep_file_test, axis=0)    
covarep_indexes = np.array(covarep_indexes)
covarep_indexes_for_sum = covarep_indexes[1:]-covarep_indexes[:-1]
covarep_indexes_for_sum = np.reshape(covarep_indexes_for_sum, (-1,1))

#print("2D indexes for sum:", covarep_indexes_for_sum)
#print("2D indexes for sum shape:", covarep_indexes_for_sum.shape)
#print("2D indexes for sum shape -5:", covarep_indexes_for_sum[:-5].shape)


#test = np.array(test_PHQ8B)
#print("is au any na:", pd.isna(au_file).any())
#print("is au test any na:", pd.isna(au_file_test).any())
scaler = MinMaxScaler()  
#Normalize train features and apply on test
covarep_file_transformed = scaler.fit_transform(covarep_file)  
covarep_file_test_transformed = scaler.transform(covarep_file_test) 

print("training RFC Model...")
#Training a predicting Randomn Forest classifier
regr = RandomForestClassifier(max_depth=2, random_state=0)              
trainregr = regr.fit(covarep_file_transformed, PHQ_score_array) 
joblib.dump(regr, "RFC Model COVAREP") 
#regr2 = joblib.load("RFC Model 3D Points")
print("is covarep test any nan:", np.nan_to_num(covarep_file_test_transformed))
print("is covarep test any inf:", np.nan_to_num(covarep_file_test_transformed))
print("covarep file test transformed:", covarep_file_test_transformed)



predictionRegr = regr.predict(np.nan_to_num(covarep_file_test_transformed))
print("Prediction RFC ",predictionRegr)
print("Prediction RFC shape",predictionRegr.shape)
print("Prediction 3D Features indexes shape",covarep_indexes[:-1].shape)
print("Prediction 3D Features shape", covarep_file_test_transformed[:-1].shape)

predictionRegrSum = np.add.reduceat(predictionRegr,covarep_indexes[:-1],0).reshape(-1,1)
print("Prediction RFC Sum shape",predictionRegrSum.shape)
predictionRegrAVG = predictionRegrSum/covarep_indexes_for_sum
predictionRegr_3BT = est.transform(predictionRegrAVG)
predictionRegr_2BT = est2.transform(predictionRegrAVG)
predictprobRFC = regr.predict_proba(np.nan_to_num(covarep_file_test))
print("---RFC Training and Prediction Done :)---")

print("############Covarep Features Metrics############", file=open('output.txt', 'a'))
print("############RFC Metrics############", file=open('output.txt', 'a'))

rfcauc2instances = np.add.reduceat(predictprobRFC, [ 0., 10., 20.][:-1],1)
rfcauc2sum = np.add.reduceat(rfcauc2instances,covarep_indexes[:-1],0)
rfcauc2 = rfcauc2sum/covarep_indexes_for_sum
rfcauc2 = rfcauc2[:,1]
rfcauc3instances = np.add.reduceat(predictprobRFC, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
rfcauc3sum = np.add.reduceat(rfcauc3instances,covarep_indexes[:-1],0)
rfcauc3 = rfcauc3sum/covarep_indexes_for_sum
print("rfcauc3:", rfcauc3)
print("rfcauc2:", rfcauc2)

regrROC3rfc = roc_auc_score(groundtruth, rfcauc3, multi_class='ovr')
regrROC2rfc = roc_auc_score(groundtruth2, rfcauc2)
accuracy3regr = accuracy_score(groundtruth, predictionRegr_3BT)
accuracy2regr = accuracy_score(groundtruth2, predictionRegr_2BT)
f13regr = f1_score(groundtruth, predictionRegr_3BT, average='macro')
f12regr = f1_score(groundtruth2, predictionRegr_2BT)
confusionmatrix3bins = confusion_matrix(groundtruth, predictionRegr_3BT).ravel()
tn2regr, fp2regr, fn2regr, tp2regr = confusion_matrix(groundtruth2, predictionRegr_2BT).ravel()
print("ROC AUC(3bins):", regrROC3rfc, file=open('output.txt', 'a'))
print("ROC AUC(2bins):", regrROC2rfc, file=open('output.txt', 'a'))
print("Accuracy (3bins):", accuracy3regr, file=open('output.txt', 'a'))
print("Accuracy (2bins):", accuracy2regr, file=open('output.txt', 'a'))
print("F1-Score (3bins):", f13regr, file=open('output.txt', 'a'))
print("F1-Score (2bins):", f12regr, file=open('output.txt', 'a'))
print("Confusion Matrix (3bins):", confusionmatrix3bins, file=open('output.txt', 'a'))
print("True Positive (2bins):", tp2regr, file=open('output.txt', 'a'))
print("Flase Positive (2bins):", fp2regr, file=open('output.txt', 'a'))
print("True Negative (2bins):", tn2regr, file=open('output.txt', 'a'))
print("False Negative (2bins):", fn2regr, file=open('output.txt', 'a'))



print("XGBoost Training and Prediction...")
#Training and Predicting XGBoost
xgb_model = xgb.XGBClassifier(tree_method = 'gpu_hist')
trainxgb = xgb_model.fit(covarep_file_transformed, PHQ_score_array)
joblib.dump(xgb_model, "XGB model COVAREP") 
#xgb_model2 = joblib.load("XGB Model COVAREP")
print("XGB model saved")

predictionXGB = xgb_model.predict(np.nan_to_num(covarep_file_test_transformed))
predictionXGBsum = np.add.reduceat(predictionXGB,covarep_indexes[:-1],0).reshape(-1,1)
predictionXGBAVG = predictionXGBsum/covarep_indexes_for_sum
predictionXGB_3BT = est.transform(predictionXGBAVG)
predictionXGB_2BT = est2.transform(predictionXGBAVG)
predictprobXGB = xgb_model.predict_proba(np.nan_to_num(covarep_file_test))
print("---XGBoost Training and Prediction Done :)---")

#XGB metrics
print("############XGBoost Metrics############", file=open('output.txt', 'a'))
xgbauc2instances = np.add.reduceat(predictprobXGB, [ 0., 10., 20.][:-1],1)
xgbauc2sum = np.add.reduceat(xgbauc2instances,covarep_indexes[:-1],0)
xgbauc2 = xgbauc2sum/covarep_indexes_for_sum
xgbauc2 = xgbauc2[:,1]
xgbauc3instances = np.add.reduceat(predictprobXGB, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
xgbauc3sum = np.add.reduceat(xgbauc3instances,covarep_indexes[:-1],0)
xgbauc3 = xgbauc3sum/covarep_indexes_for_sum

regrROC3xgb = roc_auc_score(groundtruth, xgbauc3,  multi_class='ovr')
regrROC2xgb = roc_auc_score(groundtruth2, xgbauc2)
accuracy3xgb = accuracy_score(groundtruth, predictionXGB_3BT)
accuracy2xgb = accuracy_score(groundtruth2, predictionXGB_2BT)
f13xgb = f1_score(groundtruth, predictionXGB_3BT, average='macro')
f12xgb = f1_score(groundtruth2, predictionXGB_2BT)
xgbconfusionmatrix3bins = confusion_matrix(groundtruth, predictionXGB_3BT).ravel()
tn2xgb, fp2xgb, fn2xgb, tp2xgb = confusion_matrix(groundtruth2, predictionXGB_2BT).ravel()
print("ROC AUC(3bins):", regrROC3xgb, file=open('output.txt', 'a'))
print("ROC AUC(2bins):", regrROC2xgb, file=open('output.txt', 'a'))
print("Accuracy (3bins):", accuracy3xgb, file=open('output.txt', 'a'))
print("Accuracy (2bins):", accuracy2xgb, file=open('output.txt', 'a'))
print("F1-Score (3bins):", f13xgb, file=open('output.txt', 'a'))
print("F1-Score (2bins):", f12xgb, file=open('output.txt', 'a'))
print("Confusion Matrix (3bins):", xgbconfusionmatrix3bins, file=open('output.txt', 'a'))
print("True Positive (2bins):", tp2xgb, file=open('output.txt', 'a'))
print("Flase Positive (2bins):", fp2xgb, file=open('output.txt', 'a'))
print("True Negative (2bins):", tn2xgb, file=open('output.txt', 'a'))
print("False Negative (2bins):", fn2xgb, file=open('output.txt', 'a'))


print("Naive Bayes Training and Prediction...")
gnb = GaussianNB()
traingnb = gnb.fit(covarep_file_transformed, PHQ_score_array)
joblib.dump(gnb, "GNB model COVAREP") 
#gnb2 = joblib.load("GNB Model COVAREP")
print("GNB model saved")
predictionGNB = gnb.predict(np.nan_to_num(covarep_file_test_transformed))
predictionGNBSum = np.add.reduceat(predictionGNB,covarep_indexes[:-1],0).reshape(-1,1)
predictionGNBAVG = predictionGNBSum/covarep_indexes_for_sum
predictionGNB_3BT = est.transform(predictionGNBAVG)
predictionGNB_2BT = est2.transform(predictionGNBAVG)
predictprobGNB = gnb.predict_proba(np.nan_to_num(covarep_file_test))

gnbauc2instances = np.add.reduceat(predictprobGNB, [ 0., 10., 20.][:-1],1)
gnbauc2sum = np.add.reduceat(gnbauc2instances,covarep_indexes[:-1],0)
gnbauc2 = gnbauc2sum/covarep_indexes_for_sum
gnbauc2 = gnbauc2[:,1]
gnbauc3instances = np.add.reduceat(predictprobGNB, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
gnbauc3sum = np.add.reduceat(gnbauc3instances,covarep_indexes[:-1],0)
gnbauc3 = gnbauc3sum/covarep_indexes_for_sum
print("---Naive Bayes Training and Prediction Done :)---")

#metrics for GNB
print("############Naive Bayes Metrics############", file=open('output.txt', 'a'))
regrROC3gnb = roc_auc_score(groundtruth, gnbauc3,  multi_class='ovr')
regrROC2gnb = roc_auc_score(groundtruth2, gnbauc2)
accuracygnb = accuracy_score(groundtruth, predictionGNB_3BT)
accuracy2gnb = accuracy_score(groundtruth2, predictionGNB_2BT)
f13gnb = f1_score(groundtruth, predictionGNB_3BT, average='macro')
f12gnb = f1_score(groundtruth2, predictionGNB_2BT)
gnbconfusionmatrix3bins = confusion_matrix(groundtruth, predictionGNB_3BT).ravel()
tn2gnb, fp2gnb, fn2gnb, tp2gnb = confusion_matrix(groundtruth2, predictionGNB_2BT).ravel()
print("ROC AUC(3bins):", regrROC3gnb, file=open('output.txt', 'a'))
print("ROC AUC(2bins):", regrROC2gnb, file=open('output.txt', 'a'))
print("Accuracy (3bins):", accuracygnb, file=open('output.txt', 'a'))
print("Accuracy (2bins):", accuracy2gnb, file=open('output.txt', 'a'))
print("F1-Score (3bins):", f13gnb, file=open('output.txt', 'a'))
print("F1-Score (2bins):", f12gnb, file=open('output.txt', 'a'))
print("Confusion Matrix (3bins):", gnbconfusionmatrix3bins, file=open('output.txt', 'a'))
print("True Positive (2bins):", tp2gnb, file=open('output.txt', 'a'))
print("Flase Positive (2bins):", fp2gnb, file=open('output.txt', 'a'))
print("True Negative (2bins):", tn2gnb, file=open('output.txt', 'a'))
print("False Negative (2bins):", fn2gnb, file=open('output.txt', 'a'))


print("SVM Training and Prediction...")

clf=svm.LinearSVC()
trainclf = clf.fit(covarep_file_transformed, PHQ_score_array)
joblib.dump(clf, "SVM Model COVAREP") 
#clf2 = joblib.load("SVM Model 2D")
print("SVM model saved")
predictionSVM = clf.predict(np.nan_to_num(covarep_file_test_transformed))  
predictionSVMSUM =  np.add.reduceat(predictionSVM,covarep_indexes[:-1],0).reshape(-1,1)
predictionSVMAVG = predictionSVMSUM/covarep_indexes_for_sum
predictionSVM_3BT = est.transform(predictionSVMAVG)
predictionSVM_2BT = est2.transform(predictionSVMAVG)
predictprobSVM = clf._predict_proba_lr(np.nan_to_num(covarep_file_test))
print("---SVM Training and Prediction Done :)---")

print("############3D Features SVM Metrics############", file=open('output.txt', 'a'))
svmauc2instances = np.add.reduceat(predictprobSVM, [ 0., 10., 20.][:-1],1)
svmauc2sum = np.add.reduceat(svmauc2instances, covarep_indexes[:-1],0)
svmauc2 = svmauc2sum/covarep_indexes_for_sum
svmauc2 = svmauc2[:,1]
svmauc3instances = np.add.reduceat(predictprobSVM, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
svmauc3sum = np.add.reduceat(svmauc3instances,covarep_indexes[:-1],0)
svmauc3 = svmauc3sum/covarep_indexes_for_sum

regrROC3svm = roc_auc_score(groundtruth, svmauc3, multi_class='ovr')
regrROC2svm = roc_auc_score(groundtruth2, svmauc2)
accuracy3svm = accuracy_score(groundtruth, predictionSVM_3BT)
accuracy2svm = accuracy_score(groundtruth2, predictionSVM_2BT)
f13svm = f1_score(groundtruth, predictionSVM_3BT, average='macro')
f12svm = f1_score(groundtruth2, predictionSVM_2BT)
svmconfusionmatrix3bins = confusion_matrix(groundtruth, predictionSVM_3BT).ravel()
tn2svm, fp2svm, fn2svm, tp2svm = confusion_matrix(groundtruth2, predictionSVM_2BT).ravel()
print("ROC AUC(3bins):", regrROC3svm, file=open('output.txt', 'a'))
print("ROC AUC(2bins):", regrROC2svm, file=open('output.txt', 'a'))
print("Accuracy (3bins):", accuracy3svm, file=open('output.txt', 'a'))
print("Accuracy (2bins):", accuracy2svm, file=open('output.txt', 'a'))
print("F1-Score (3bins):", f13svm, file=open('output.txt', 'a'))
print("F1-Score (2bins):", f12svm, file=open('output.txt', 'a'))
print("Confusion Matrix (3bins):", svmconfusionmatrix3bins, file=open('output.txt', 'a'))
print("True Positive (2bins):", tp2svm, file=open('output.txt', 'a'))
print("Flase Positive (2bins):", fp2svm, file=open('output.txt', 'a'))
print("True Negative (2bins):", tn2svm, file=open('output.txt', 'a'))
print("False Negative (2bins):", fn2svm, file=open('output.txt', 'a'))


print("Finished lets GOOOOOOOOOOOOOO")
    
