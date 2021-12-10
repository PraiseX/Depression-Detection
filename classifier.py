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
import xgboost as xgb                         
import joblib  

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


#print("Groundtruth(3bins):", est.transform(test_PHQ8B))
print("Groundtruth(3bins(reshaped)):", groundtruth)
print("Groundtruth(2bins(reshaped)):", groundtruth2)


#print("groundtrush is:", groundtruth)

#train_PHQ8B_float = pd.to_numeric(train_PHQ8B,downcast='float')

# print("type train_PHQ8B_float: ", type(train_PHQ8B_float))


au_file = [] 
au_file_test = []

PHQ_score = []
PHQ_score_test = []

AU_indexes = [0]
au_file_len_idx = 0

print("---loading AU Files----")


#Load train and test features and labels
#make sure all the participants from the training split are in folder
for idx, train_u in enumerate(train_user_id):
    #depending on the file use different seperators
    au_file_tmp = pd.read_csv(path+'{}_CLNF_AUs.txt'.format(train_u), sep = ', ')
    #au_file_tmp_mean = au_file_tmp.iloc[:, -22:].mean(axis=0)
    #au_file_tmp_std = au_file_tmp.iloc[:, -22:].std(axis=0)
    #print(type(au_file_tmp))
    au_file_len = len(au_file_tmp)
    au_score = train_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_tmp = [au_score]*au_file_len
    #print("AU score:", au_score)
    PHQ_score.extend(PHQ_score_tmp)

    #print("is au tmp any na:", pd.isna(au_file_tmp).any())
     # calculate the std of each column
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    #concate  = pd.concat([au_file_tmp_mean, au_file_tmp_std], axis=0)
    #au_file.append(concate)
    au_file.append(au_file_tmp[['confidence', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']])  
                                                               


for idx, test_u in enumerate(test_user_id): 
    #depending on the file use different seperators
    au_file_test_tmp = pd.read_csv(path+'{}_CLNF_AUs.txt'.format(test_u), sep = ', ')
    au_file_test_tmp_mean = au_file_test_tmp.iloc[:, -22:].mean(axis=0)
    au_file_test_tmp_std = au_file_test_tmp.iloc[:, -22:].std(axis=0)
    #print("is au tmp test any na:", pd.isna(au_file_test_tmp).any())
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    au_file_len = len(au_file_test_tmp)
    au_file_len_idx += au_file_len
    AU_indexes.append(au_file_len_idx)

    au_score = test_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_test_tmp = [au_score]*au_file_len
    PHQ_score_test.extend(PHQ_score_test_tmp)
    concate  = pd.concat([au_file_test_tmp_mean, au_file_test_tmp_std], axis=1)
    #au_file_test.append(concate)
    au_file_test.append(au_file_test_tmp[['confidence', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']]) 
    # print("train_u:" , train_u) 
    # print("au_file:" , au_file)                                                             

                    
print("---done loading AU Files----")
PHQ_score_array = np.array(PHQ_score)
PHQ_score_R = np.reshape(PHQ_score_array,(-1,1))
PHQ_score_test_array = np.array(PHQ_score_test)
PHQ_score_test_R = np.reshape(PHQ_score_test_array,(-1,1))
au_file = pd.concat(au_file, axis=0)    
au_file_test = pd.concat(au_file_test, axis=0)    
AU_indexes = np.array(AU_indexes)
AU_indexes_for_sum = AU_indexes[1:]-AU_indexes[:-1]
AU_indexes_for_sum = np.reshape(AU_indexes_for_sum, (-1,1))

print("AU indexes for sum:", AU_indexes_for_sum)
#test = np.array(test_PHQ8B)
#print("is au any na:", pd.isna(au_file).any())
#print("is au test any na:", pd.isna(au_file_test).any())
scaler = MinMaxScaler()  
#Normalize train features and apply on test
au_file_transformed = scaler.fit_transform(au_file)  
au_file_test_transformed = scaler.transform(au_file_test) 


###NOTE If an error occurs after a model is trained comment out model.fit(x,y) and just uncomment modelname2 and use that predict the test data

print("Loading RFC Model...")
#Training a predicting Randomn Forest classifier
regr = RandomForestClassifier(max_depth=2, random_state=0)              
#trainregr = regr.fit(au_file_transformed, PHQ_score_array) 
#joblib.dump(regr, "RFC model") 
regr2 = joblib.load("RFC Model")
#print("Trained RFC model loaded, predicting...")
predictionRegr = regr2.predict(au_file_test_transformed) 
#print("Prediction Rfc:", predictionRegr)
#print("prediction rfc:", predictionRegr)
#print("AU_indexes minus 1", AU_indexes[:-1])  
predictionRegrSum = np.add.reduceat(predictionRegr,AU_indexes[:-1],0).reshape(-1,1)
print("Prediction RFC Sum shape",predictionRegrSum.shape)
predictionRegrAVG = predictionRegrSum/AU_indexes_for_sum
#print("Predict RFC AVG:",predictionRegrAVG)
predictionRegr_3BT = est.transform(predictionRegrAVG)
predictionRegr_2BT = est2.transform(predictionRegrAVG)
predictprobRFC = regr2.predict_proba(au_file_test)
#print("Predict Probability RFC:", predictprobRFC)
print("---RFC Training and Prediction Done :)---")
#print("############Random Forest Classifier Metrics############", file=open('output.txt', 'a'))
#metrics for regression

rfcauc2instances = np.add.reduceat(predictprobRFC, [ 0., 10., 20.][:-1],1)
#print("rfc auc 2 instances shape", rfcauc2instances.shape)
#print("rfc auc 2 instances ", rfcauc2instances)

rfcauc2sum = np.add.reduceat(rfcauc2instances,AU_indexes[:-1],0)
#print("rfc auc 2 sum shape", rfcauc2sum.shape)
#print("rfc auc 2 sum", rfcauc2sum)

rfcauc2 = rfcauc2sum/AU_indexes_for_sum
rfcauc2 = rfcauc2

#rfcauc2 = rfcauc2[:,:-1]

rfcauc3instances = np.add.reduceat(predictprobRFC, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
rfcauc3sum = np.add.reduceat(rfcauc3instances,AU_indexes[:-1],0)
rfcauc3 = rfcauc3sum/AU_indexes_for_sum

print("groundtruth2(y_true) shape:", groundtruth2.shape)
#print("rfcauc3(y_score):",rfcauc3)
print("rfcauc2(y_score) shape:",rfcauc2.shape)

regrROC3rfc = roc_auc_score(groundtruth, rfcauc3, multi_class='ovr')
regrROC2rfc = roc_auc_score(groundtruth2, rfcauc2, multi_class='ovr')
accuracy3regr = accuracy_score(groundtruth, predictionRegr_3BT)
accuracy2regr = accuracy_score(groundtruth2, predictionRegr_2BT)
f13regr = f1_score(groundtruth, predictionRegr_3BT)
f12regr = f1_score(groundtruth2, predictionRegr_2BT)
tn3regr, fp3regr, fn3regr, tp3regr = confusion_matrix(groundtruth, predictionRegr_3BT).ravel()
tn2regr, fp2regr, fn2regr, tp2regr = confusion_matrix(groundtruth2, predictionRegr_2BT).ravel()
print("ROC AUC(3bins):", regrROC3rfc, file=open('output.txt', 'a'))
print("ROC AUC(2bins):", regrROC2rfc, file=open('output.txt', 'a'))
print("Accuracy (3bins):", accuracy3regr, file=open('output.txt', 'a'))
print("Accuracy (2bins):", accuracy2regr, file=open('output.txt', 'a'))
print("F1-Score (3bins):", f13regr, file=open('output.txt', 'a'))
print("F1-Score (2bins):", f12regr, file=open('output.txt', 'a'))
print("True Positive (3bins):", tp3regr, file=open('output.txt', 'a'))
print("Flase Positive (3bins):", fp3regr, file=open('output.txt', 'a'))
print("True Negative (3bins):", tn3regr, file=open('output.txt', 'a'))
print("False Negative (3bins):", fn3regr, file=open('output.txt', 'a'))

print("SVM Training and Prediction...")
clf=svm.SVC()
trainclf = clf.fit(au_file_transformed, PHQ_score_array)
joblib.dump(clf, "SVM model") 
#clf2 = joblib.load("SVM Model")
print("SVM model saved")
predictionSVM = clf.predict(au_file_test_transformed)   
predictionSVMAVG = np.add.reduceat(predictionSVM,AU_indexes[:-1],0).reshape(-1,1)
predictionSVM_3BT = est.transform(predictionSVMAVG)
predictionSVM_2BT = est2.transform(predictionSVMAVG)
predictprobSVM = clf.predict_proba(au_file_test)
print("---SVM Training and Prediction Done :)---")

print("############SVM Metrics############", file=open('output.txt', 'a'))
svmauc2 = np.add.reduceat(predictprobSVM, est2.bin_edges_,1)[:,:-1]
svmauc3 = np.add.reduceat(predictprobSVM, est.bin_edges_,1)[:,:-1]
regrROC3svm = roc_auc_score(groundtruth, svmauc3, multi_class='ovr')
regrROC2svm = roc_auc_score(groundtruth2, svmauc2)
accuracy3svm = accuracy_score(groundtruth, predictionRegr_3BT)
accuracy2svm = accuracy_score(groundtruth2, predictionRegr_2BT)
f13svm = f1_score(groundtruth, predictionRegr_3BT)
f12svm = f1_score(groundtruth2, predictionRegr_2BT)
tn3svm, fp3svm, fn3svm, tp3svm = confusion_matrix(groundtruth, predictionRegr_3BT).ravel()
tn2svm, fp2svm, fn2svm, tp2svm = confusion_matrix(groundtruth2, predictionRegr_2BT).ravel()
print("ROC AUC(3bins):", regrROC3svm, file=open('output.txt', 'a'))
print("ROC AUC(2bins):", regrROC2svm, file=open('output.txt', 'a'))
print("Accuracy (3bins):", accuracy3svm, file=open('output.txt', 'a'))
print("Accuracy (2bins):", accuracy2svm, file=open('output.txt', 'a'))
print("F1-Score (3bins):", f13svm, file=open('output.txt', 'a'))
print("F1-Score (2bins):", f12svm, file=open('output.txt', 'a'))
print("True Positive (3bins):", tp3svm, file=open('output.txt', 'a'))
print("Flase Positive (3bins):", fp3svm, file=open('output.txt', 'a'))
print("True Negative (3bins):", tn3svm, file=open('output.txt', 'a'))
print("False Negative (3bins):", fn3svm, file=open('output.txt', 'a'))
print("True Positive (2bins):", tp2svm, file=open('output.txt', 'a'))
print("Flase Positive (2bins):", fp2svm, file=open('output.txt', 'a'))
print("True Negative (2bins):", tn2svm, file=open('output.txt', 'a'))
print("False Negative (2bins):", fn2svm, file=open('output.txt', 'a'))


print("XGBoost Training and Prediction...")
#Training and Predicting XGBoost
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
trainxgb = xgb_model.fit(au_file_transformed, PHQ_score_array)
joblib.dump(xgb_model, "XGB model") 
#xgb_model2 = joblib.load("XGB Model")
print("XGB model saved")
predictionXGB = xgb_model.predict(au_file_test_transformed)
predictionXGBAVG = np.add.reduceat(predictionXGB,AU_indexes[:-1],0).reshape(-1,1)
predictionXGB_3BT = est.transform(predictionXGBAVG)
predictionXGB_2BT = est2.transform(predictionXGBAVG)
predictprobXGB = xgb_model.classes_
print("---XGBoost Training and Prediction Done :)---")

#XGB metrics
print("############XGBoost Metrics############", file=open('output.txt', 'a'))
xgbauc2 = np.add.reduceat(predictprobXGB, est2.bin_edges_,1)[:,:-1]
xgbauc3 = np.add.reduceat(predictprobXGB, est.bin_edges_,1)[:,:-1]
regrROC3xgb = roc_auc_score(groundtruth, xgbauc3,  multi_class='ovr')
regrROC2xgb = roc_auc_score(groundtruth2, xgbauc2)
accuracy3xgb = accuracy_score(groundtruth, predictionRegr_3BT)
accuracy2xgb = accuracy_score(groundtruth2, predictionRegr_2BT)
f13xgb = f1_score(groundtruth, predictionRegr_3BT)
f12xgb = f1_score(groundtruth2, predictionRegr_2BT)
tn3xgb, fp3xgb, fn3xgb, tp3xgb = confusion_matrix(groundtruth, predictionRegr_3BT).ravel()
tn2xgb, fp2xgb, fn2xgb, tp2xgb = confusion_matrix(groundtruth2, predictionRegr_2BT).ravel()
print("ROC AUC(3bins):", regrROC3xgb, file=open('output.txt', 'a'))
print("ROC AUC(2bins):", regrROC2xgb, file=open('output.txt', 'a'))
print("Accuracy (3bins):", accuracy3xgb, file=open('output.txt', 'a'))
print("Accuracy (2bins):", accuracy2xgb, file=open('output.txt', 'a'))
print("F1-Score (3bins):", f13xgb, file=open('output.txt', 'a'))
print("F1-Score (2bins):", f12xgb, file=open('output.txt', 'a'))
print("True Positive (3bins):", tp3xgb, file=open('output.txt', 'a'))
print("Flase Positive (3bins):", fp3xgb, file=open('output.txt', 'a'))
print("True Negative (3bins):", tn3xgb, file=open('output.txt', 'a'))
print("False Negative (3bins):", fn3xgb, file=open('output.txt', 'a'))
print("True Positive (2bins):", tp2xgb, file=open('output.txt', 'a'))
print("Flase Positive (2bins):", fp2xgb, file=open('output.txt', 'a'))
print("True Negative (2bins):", tn2xgb, file=open('output.txt', 'a'))
print("False Negative (2bins):", fn2xgb, file=open('output.txt', 'a'))


print("Naive Bayes Training and Prediction...")
gnb = GaussianNB()
traingnb = gnb.fit(au_file_transformed, PHQ_score_array)
joblib.dump(gnb, "GNB model") 
#gnb2 = joblib.load("GNB Model")
print("GNB model saved")
predictionGNB = gnb.predict(au_file_test_transformed)
predictionGNBAVG = np.add.reduceat(predictionGNB,AU_indexes[:-1],0).reshape(-1,1)
predictionGNB_3BT = est.transform(predictionGNBAVG)
predictionGNB_2BT = est2.transform(predictionGNBAVG)
predictprobGNB = gnb.predict_proba(au_file_test)
gnbauc2 = np.add.reduceat(predictprobGNB, est2.bin_edges_,1)[:,:-1]
gnbauc3 = np.add.reduceat(predictprobGNB, est.bin_edges_,1)[:,:-1]
print("---Naive Bayes Training and Prediction Done :)---")

#metrics for GNB
print("############Naive Bayes Metrics############", file=open('output.txt', 'a'))
regrROC3gnb = roc_auc_score(groundtruth, gnbauc3,  multi_class='ovr')
regrROC2gnb = roc_auc_score(groundtruth2, gnbauc2)
accuracygnb = accuracy_score(groundtruth, predictionGNB_3BT)
accuracy2gnb = accuracy_score(groundtruth2, predictionGNB_2BT)
f13gnb = f1_score(groundtruth, predictionGNB_3BT)
f12gnb = f1_score(groundtruth2, predictionGNB_2BT)
tn3gnb, fp3gnb, fn3gnb, tp3gnb = confusion_matrix(groundtruth, predictionGNB_3BT).ravel()
tn2gnb, fp2gnb, fn2gnb, tp2gnb = confusion_matrix(groundtruth2, predictionGNB_2BT).ravel()
print("ROC AUC(3bins):", regrROC3gnb, file=open('output.txt', 'a'))
print("ROC AUC(2bins):", regrROC2gnb, file=open('output.txt', 'a'))
print("Accuracy (3bins):", accuracy3xgb, file=open('output.txt', 'a'))
print("Accuracy (2bins):", accuracy2gnb, file=open('output.txt', 'a'))
print("F1-Score (3bins):", f13gnb, file=open('output.txt', 'a'))
print("F1-Score (2bins):", f12gnb, file=open('output.txt', 'a'))
print("True Positive (3bins):", tp3gnb, file=open('output.txt', 'a'))
print("Flase Positive (3bins):", fp3gnb, file=open('output.txt', 'a'))
print("True Negative (3bins):", tn3gnb, file=open('output.txt', 'a'))
print("False Negative (3bins):", fn3gnb, file=open('output.txt', 'a'))
print("True Positive (2bins):", tp2gnb, file=open('output.txt', 'a'))
print("Flase Positive (2bins):", fp2gnb, file=open('output.txt', 'a'))
print("True Negative (2bins):", tn2gnb, file=open('output.txt', 'a'))
print("False Negative (2bins):", fn2gnb, file=open('output.txt', 'a'))


print("Finished lets GOOOOOOOOOOOOOO")
























                                                                                
 