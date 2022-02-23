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
path = "/home/peteng/data/Facial Features/Pose/"
filenames = glob.glob(path + "\*.csv")

#replace with appropriate path
train_list = pd.read_csv('/home/peteng/data/Depression-Detection/train_split_Depression_AVEC2017.csv')                                                         
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


test_list = pd.read_csv('/home/peteng/data/Depression-Detection/dev_split_Depression_AVEC2017.csv')                                                         
test_user_id = test_list['Participant_ID']                           
test_PHQ8B = test_list[['PHQ8_Score']] 
groundtruth = est.transform(np.array(test_PHQ8B)).flatten()
groundtruth2 = est2.transform(np.array(test_PHQ8B)).flatten()

pose_file = [] 
pose_file_test = []

PHQ_score = []
PHQ_score_test = []

pose_indexes = [0]
pose_file_len_idx = 0

print("---loading pose Files----")

for idx, train_u in enumerate(train_user_id):
    #depending on the file use different seperators
    pose_file_tmp = pd.read_csv(path+'{}_CLNF_pose.txt'.format(train_u), sep = ', ')
    #au_file_tmp_mean = au_file_tmp.iloc[:, -22:].mean(axis=0)
    #au_file_tmp_std = au_file_tmp.iloc[:, -22:].std(axis=0)
    #print(type(au_file_tmp))
    pose_file_len = len(pose_file_tmp)
    pose_score = train_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_tmp = [pose_score]*pose_file_len
    #print("AU score:", au_score)
    PHQ_score.extend(PHQ_score_tmp)

    #print("is au tmp any na:", pd.isna(au_file_tmp).any())
     # calculate the std of each column
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    #concate  = pd.concat([au_file_tmp_mean, au_file_tmp_std], axis=0)
    #au_file.append(concate)
    pose_file.append(pose_file_tmp[['confidence', 'success', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']])



for idx, test_u in enumerate(test_user_id): 
    #depending on the file use different seperators
    pose_file_test_tmp = pd.read_csv(path+'{}_CLNF_pose.txt'.format(test_u), sep = ', ')
    pose_file_test_tmp_mean = pose_file_test_tmp.iloc[:, -8:].mean(axis=0)
    pose_file_test_tmp_std = pose_file_test_tmp.iloc[:, -8:].std(axis=0)
    #print("is au tmp test any na:", pd.isna(au_file_test_tmp).any())
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    pose_file_len = len(pose_file_test_tmp)
    pose_file_len_idx += pose_file_len
    pose_indexes.append(pose_file_len_idx)

    pose_score = test_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_test_tmp = [pose_score]*pose_file_len
    PHQ_score_test.extend(PHQ_score_test_tmp)
    concate  = pd.concat([pose_file_test_tmp_mean, pose_file_test_tmp_std], axis=1)
    #au_file_test.append(concate)
    pose_file_test.append(pose_file_test_tmp[['confidence', 'success', 'Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']])
    # print("train_u:" , train_u) 
    # print("au_file:" , au_file)                                                             

print("---done loading pose Files----")
PHQ_score_array = np.array(PHQ_score)
PHQ_score_R = np.reshape(PHQ_score_array,(-1,1))
PHQ_score_test_array = np.array(PHQ_score_test)
PHQ_score_test_R = np.reshape(PHQ_score_test_array,(-1,1))
pose_file = pd.concat(pose_file, axis=0)    
pose_file_test = pd.concat(pose_file_test, axis=0)    
pose_indexes = np.array(pose_indexes)
pose_indexes_for_sum = pose_indexes[1:]-pose_indexes[:-1]
pose_indexes_for_sum = np.reshape(pose_indexes_for_sum, (-1,1))

#print("2D indexes for sum:", pose_indexes_for_sum)
#print("2D indexes for sum shape:", pose_indexes_for_sum.shape)
#print("2D indexes for sum shape -5:", pose_indexes_for_sum[:-5].shape)


#test = np.array(test_PHQ8B)
#print("is au any na:", pd.isna(au_file).any())
#print("is au test any na:", pd.isna(au_file_test).any())
scaler = MinMaxScaler()  
#Normalize train features and apply on test
pose_file_transformed = scaler.fit_transform(pose_file)  
pose_file_test_transformed = scaler.transform(pose_file_test) 

print("training RFC Model...")
#Training a predicting Randomn Forest classifier
parameters = [{'n_estimators': [10,100,400], 'max_depth': [None, 7], 'min_samples_split': [2,5]}]

regr = GridSearchCV(RandomForestClassifier(),param_grid=parameters,scoring='accuracy')            
trainregr = regr.fit(pose_file_transformed, PHQ_score_array) 
joblib.dump(regr, "RFC Model pose 2") 
#regr2 = joblib.load("RFC Model 3D Points")
#print("is 2d test any nan:", np.nan_to_num(pose_file_test_transformed))
#print("is 2d test any inf:", np.nan_to_num(pose_file_test_transformed))
#print("2d file test transformed:", pose_file_test_transformed)



predictionRegr = regr.predict(np.nan_to_num(pose_file_test_transformed))
#print("Prediction RFC ",predictionRegr)
print("Prediction RFC shape",predictionRegr.shape)
print("Prediction 3D Features indexes shape",pose_indexes[:-1].shape)
print("Prediction 3D Features shape", pose_file_test_transformed[:-1].shape)

predictionRegrSum = np.add.reduceat(predictionRegr,pose_indexes[:-1],0).reshape(-1,1)
#print("Prediction RFC Sum shape",predictionRegrSum.shape)
predictionRegrAVG = predictionRegrSum/pose_indexes_for_sum
predictionRegr_3BT = est.transform(predictionRegrAVG)
predictionRegr_2BT = est2.transform(predictionRegrAVG)
predictprobRFC = regr.predict_proba(np.nan_to_num(pose_file_test_transformed))
print("---RFC Training and Prediction Done :)---")

f=open(r"poseOutput.txt", "a")
f.write("############Pose Features Metrics##########")
f.write("############RFC Metrics############")
f.write("best parameters:", regr.best_params_ )

rfcauc2instances = np.add.reduceat(predictprobRFC, [ 0., 10., 20.][:-1],1)
rfcauc2sum = np.add.reduceat(rfcauc2instances,pose_indexes[:-1],0)
rfcauc2 = rfcauc2sum/pose_indexes_for_sum
rfcauc2 = rfcauc2[:,1]
rfcauc3instances = np.add.reduceat(predictprobRFC, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
rfcauc3sum = np.add.reduceat(rfcauc3instances,pose_indexes[:-1],0)
rfcauc3 = rfcauc3sum/pose_indexes_for_sum
#print("rfcauc3:", rfcauc3)
#print("rfcauc2:", rfcauc2)

regrROC3rfc = roc_auc_score(groundtruth, rfcauc3, multi_class='ovr')
regrROC2rfc = roc_auc_score(groundtruth2, rfcauc2)
accuracy3regr = accuracy_score(groundtruth, predictionRegr_3BT)
accuracy2regr = accuracy_score(groundtruth2, predictionRegr_2BT)
f13regr = f1_score(groundtruth, predictionRegr_3BT, average='macro')
f12regr = f1_score(groundtruth2, predictionRegr_2BT)
confusionmatrix3bins = confusion_matrix(groundtruth, predictionRegr_3BT).ravel()
tn2regr, fp2regr, fn2regr, tp2regr = confusion_matrix(groundtruth2, predictionRegr_2BT).ravel()
f.write("ROC AUC(3bins):", regrROC3rfc)
f.write("ROC AUC(2bins):", regrROC2rfc)
f.write("Accuracy (3bins):", accuracy3regr)
f.write("Accuracy (2bins):", accuracy2regr)
f.write("F1-Score (3bins):", f13regr)
f.write("F1-Score (2bins):", f12regr)
f.write("Confusion Matrix (3bins):", confusionmatrix3bins)
f.write("True Positive (2bins):", tp2regr)
f.write("Flase Positive (2bins):", fp2regr)
f.write("True Negative (2bins):", tn2regr)
f.write("False Negative (2bins):", fn2regr)


print("XGBoost Training and Prediction...")
# Training and Predicting XGBoost
xgb_model = xgb.XGBClassifier(learning_rate=0.01,gamma = 5, max_depth=4,min_child_weight=1, subsample=0.7, colsample_bytree=0.6)
trainxgb = xgb_model.fit(pose_file_transformed, PHQ_score_array)
joblib.dump(xgb_model, "XGB model pose 2") 
#xgb_model2 = joblib.load("XGB Model 2D")
print("XGB model saved")

predictionXGB = xgb_model.predict(np.nan_to_num(pose_file_test_transformed))
predictionXGBsum = np.add.reduceat(predictionXGB,pose_indexes[:-1],0).reshape(-1,1)
predictionXGBAVG = predictionXGBsum/pose_indexes_for_sum
predictionXGB_3BT = est.transform(predictionXGBAVG)
predictionXGB_2BT = est2.transform(predictionXGBAVG)
predictprobXGB = xgb_model.predict_proba(np.nan_to_num(pose_file_test_transformed))
print("---XGBoost Training and Prediction Done :)---")

# XGB metrics
xgbauc2instances = np.add.reduceat(predictprobXGB, [ 0., 10., 20.][:-1],1)
xgbauc2sum = np.add.reduceat(xgbauc2instances,pose_indexes[:-1],0)
xgbauc2 = xgbauc2sum/pose_indexes_for_sum
xgbauc2 = xgbauc2[:,1]
xgbauc3instances = np.add.reduceat(predictprobXGB, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
xgbauc3sum = np.add.reduceat(xgbauc3instances,pose_indexes[:-1],0)
xgbauc3 = xgbauc3sum/pose_indexes_for_sum

regrROC3xgb = roc_auc_score(groundtruth, xgbauc3,  multi_class='ovr')
regrROC2xgb = roc_auc_score(groundtruth2, xgbauc2)
accuracy3xgb = accuracy_score(groundtruth, predictionXGB_3BT)
accuracy2xgb = accuracy_score(groundtruth2, predictionXGB_2BT)
f13xgb = f1_score(groundtruth, predictionXGB_3BT, average='macro')
f12xgb = f1_score(groundtruth2, predictionXGB_2BT)
xgbconfusionmatrix3bins = confusion_matrix(groundtruth, predictionXGB_3BT).ravel()
tn2xgb, fp2xgb, fn2xgb, tp2xgb = confusion_matrix(groundtruth2, predictionXGB_2BT).ravel()
f.write("############XGBoost Metrics############")
f.write("ROC AUC(3bins):", regrROC3xgb)
f.write("ROC AUC(2bins):", regrROC2xgb)
f.write("Accuracy (3bins):", accuracy3xgb)
f.write("Accuracy (2bins):", accuracy2xgb)
f.write("F1-Score (3bins):", f13xgb)
f.write("F1-Score (2bins):", f12xgb)
f.write("Confusion Matrix (3bins):", xgbconfusionmatrix3bins)
f.write("True Positive (2bins):", tp2xgb)
f.write("Flase Positive (2bins):", fp2xgb)
f.write("True Negative (2bins):", tn2xgb)
f.write("False Negative (2bins):", fn2xgb)

print("Naive Bayes Training and Prediction...")
gnb = GaussianNB()
traingnb = gnb.fit(pose_file_transformed, PHQ_score_array)
joblib.dump(gnb, "GNB model pose 2") 
#gnb2 = joblib.load("GNB Model 2D")
print("GNB model saved")
predictionGNB = gnb.predict(np.nan_to_num(pose_file_test_transformed))
predictionGNBSum = np.add.reduceat(predictionGNB,pose_indexes[:-1],0).reshape(-1,1)
predictionGNBAVG = predictionGNBSum/pose_indexes_for_sum
predictionGNB_3BT = est.transform(predictionGNBAVG)
predictionGNB_2BT = est2.transform(predictionGNBAVG)
predictprobGNB = gnb.predict_proba(np.nan_to_num(pose_file_test_transformed))

gnbauc2instances = np.add.reduceat(predictprobGNB, [ 0., 10., 20.][:-1],1)
gnbauc2sum = np.add.reduceat(gnbauc2instances,pose_indexes[:-1],0)
gnbauc2 = gnbauc2sum/pose_indexes_for_sum
gnbauc2 = gnbauc2[:,1]
gnbauc3instances = np.add.reduceat(predictprobGNB, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
gnbauc3sum = np.add.reduceat(gnbauc3instances,pose_indexes[:-1],0)
gnbauc3 = gnbauc3sum/pose_indexes_for_sum
print("---Naive Bayes Training and Prediction Done :)---")

# metrics for GNB
regrROC3gnb = roc_auc_score(groundtruth, gnbauc3,  multi_class='ovr')
regrROC2gnb = roc_auc_score(groundtruth2, gnbauc2)
accuracygnb = accuracy_score(groundtruth, predictionGNB_3BT)
accuracy2gnb = accuracy_score(groundtruth2, predictionGNB_2BT)
f13gnb = f1_score(groundtruth, predictionGNB_3BT, average='macro')
f12gnb = f1_score(groundtruth2, predictionGNB_2BT)
gnbconfusionmatrix3bins = confusion_matrix(groundtruth, predictionGNB_3BT).ravel()
tn2gnb, fp2gnb, fn2gnb, tp2gnb = confusion_matrix(groundtruth2, predictionGNB_2BT).ravel()
f.write("############Naive Bayes Metrics############")
f.write("prior probabability:", gnb.class_prior_ )
f.write("ROC AUC(3bins):", regrROC3gnb)
f.write("ROC AUC(2bins):", regrROC2gnb)
f.write("Accuracy (3bins):", accuracygnb)
f.write("Accuracy (2bins):", accuracy2gnb)
f.write("F1-Score (3bins):", f13gnb)
f.write("F1-Score (2bins):", f12gnb)
f.write("Confusion Matrix (3bins):", gnbconfusionmatrix3bins)
f.write("True Positive (2bins):", tp2gnb)
f.write("Flase Positive (2bins):", fp2gnb)
f.write("True Negative (2bins):", tn2gnb)
f.write("False Negative (2bins):", fn2gnb)

print("SVM Training and Prediction...")

clf= SGDClassifier( loss='log', penalty= 'l1', max_iter=1000, tol=1e0)
trainclf = clf.fit(pose_file_transformed, PHQ_score_array)
joblib.dump(clf, "SVM Model pose 2") 
#clf2 = joblib.load("SVM Model 2D")
print("SVM model saved")
predictionSVM = clf.predict(np.nan_to_num(pose_file_test_transformed))  
predictionSVMSUM =  np.add.reduceat(predictionSVM,pose_indexes[:-1],0).reshape(-1,1)
predictionSVMAVG = predictionSVMSUM/pose_indexes_for_sum
predictionSVM_3BT = est.transform(predictionSVMAVG)
predictionSVM_2BT = est2.transform(predictionSVMAVG)
predictprobSVM = clf._predict_proba_lr(np.nan_to_num(pose_file_test_transformed))
print("---SVM Training and Prediction Done :)---")

svmauc2instances = np.add.reduceat(predictprobSVM, [ 0., 10., 20.][:-1],1)
svmauc2sum = np.add.reduceat(svmauc2instances, pose_indexes[:-1],0)
svmauc2 = svmauc2sum/pose_indexes_for_sum
svmauc2 = svmauc2[:,1]
svmauc3instances = np.add.reduceat(predictprobSVM, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
svmauc3sum = np.add.reduceat(svmauc3instances,pose_indexes[:-1],0)
svmauc3 = svmauc3sum/pose_indexes_for_sum

regrROC3svm = roc_auc_score(groundtruth, svmauc3, multi_class='ovr')
regrROC2svm = roc_auc_score(groundtruth2, svmauc2)
accuracy3svm = accuracy_score(groundtruth, predictionSVM_3BT)
accuracy2svm = accuracy_score(groundtruth2, predictionSVM_2BT)
f13svm = f1_score(groundtruth, predictionSVM_3BT, average='macro')
f12svm = f1_score(groundtruth2, predictionSVM_2BT)
svmconfusionmatrix3bins = confusion_matrix(groundtruth, predictionSVM_3BT).ravel()
tn2svm, fp2svm, fn2svm, tp2svm = confusion_matrix(groundtruth2, predictionSVM_2BT).ravel()
f.write("############SVM Metrics############")
f.write("ROC AUC(3bins):", regrROC3svm)
f.write("ROC AUC(2bins):", regrROC2svm)
f.write("Accuracy (3bins):", accuracy3svm)
f.write("Accuracy (2bins):", accuracy2svm)
f.write("F1-Score (3bins):", f13svm)
f.write("F1-Score (2bins):", f12svm)
f.write("Confusion Matrix (3bins):", svmconfusionmatrix3bins)
f.write("True Positive (2bins):", tp2svm)
f.write("Flase Positive (2bins):", fp2svm)
f.write("True Negative (2bins):", tn2svm)
f.write("False Negative (2bins):", fn2svm)
f.close()

print("Finished lets GOOOOOOOOOOOOOO")

#---SVM Training and Prediction Done :)---
#Finished lets GOOOOOOOOOOOOOO