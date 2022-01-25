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
path = "D:\MQP\dataset\DAICWOZ\Facial Features\\3D Features\\"
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

threeD_file = [] 
threeD_file_test = []

PHQ_score = []
PHQ_score_test = []

threeD_indexes = [0]
threeD_file_len_idx = 0

print("---loading 3D features Files----")

for idx, train_u in enumerate(train_user_id):
    #depending on the file use different seperators
    threeD_file_tmp = pd.read_csv(path+'{}_CLNF_features3D.txt'.format(train_u), sep = ', ')
    #au_file_tmp_mean = au_file_tmp.iloc[:, -22:].mean(axis=0)
    #au_file_tmp_std = au_file_tmp.iloc[:, -22:].std(axis=0)
    #print(type(au_file_tmp))
    threeD_file_len = len(threeD_file_tmp)
    threeD_score = train_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_tmp = [threeD_score]*threeD_file_len
    #print("AU score:", au_score)
    PHQ_score.extend(PHQ_score_tmp)

    #print("is au tmp any na:", pd.isna(au_file_tmp).any())
     # calculate the std of each column
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    #concate  = pd.concat([au_file_tmp_mean, au_file_tmp_std], axis=0)
    #au_file.append(concate)
    threeD_file.append(threeD_file_tmp[['confidence', 'success', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17', 'Y18', 'Y19', 'Y20', 'Y21', 'Y22', 'Y23', 'Y24', 'Y25', 'Y26', 'Y27', 'Y28', 'Y29', 'Y30', 'Y31', 'Y32', 'Y33', 'Y34', 'Y35', 'Y36', 'Y37', 'Y38', 'Y39', 'Y40', 'Y41', 'Y42', 'Y43', 'Y44', 'Y45', 'Y46', 'Y47', 'Y48', 'Y49', 'Y50', 'Y51', 'Y52', 'Y53', 'Y54', 'Y55', 'Y56', 'Y57', 'Y58', 'Y59', 'Y60', 'Y61', 'Y62', 'Y63', 'Y64', 'Y65', 'Y66', 'Y67', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20', 'Z21', 'Z22', 'Z23', 'Z24', 'Z25', 'Z26', 'Z27', 'Z28', 'Z29', 'Z30', 'Z31', 'Z32', 'Z33', 'Z34', 'Z35', 'Z36', 'Z37', 'Z38', 'Z39', 'Z40', 'Z41', 'Z42', 'Z43', 'Z44', 'Z45', 'Z46', 'Z47', 'Z48', 'Z49', 'Z50', 'Z51', 'Z52', 'Z53', 'Z54', 'Z55', 'Z56', 'Z57', 'Z58', 'Z59', 'Z60', 'Z61', 'Z62', 'Z63', 'Z64', 'Z65', 'Z66', 'Z67']])                                                          

for idx, test_u in enumerate(test_user_id): 
    #depending on the file use different seperators
    threeD_file_test_tmp = pd.read_csv(path+'{}_CLNF_features3D.txt'.format(test_u), sep = ', ')
    threeD_file_test_tmp_mean = threeD_file_test_tmp.iloc[:, -206:].mean(axis=0)
    threeD_file_test_tmp_std = threeD_file_test_tmp.iloc[:, -206:].std(axis=0)
    #print("is au tmp test any na:", pd.isna(au_file_test_tmp).any())
    # print("mean", au_file_tmp_mean)
    # print("std", au_file_tmp_std)
    threeD_file_len = len(threeD_file_test_tmp)
    threeD_file_len_idx += threeD_file_len
    threeD_indexes.append(threeD_file_len_idx)

    threeD_score = test_PHQ8B['PHQ8_Score'].iloc[idx]
    PHQ_score_test_tmp = [threeD_score]*threeD_file_len
    PHQ_score_test.extend(PHQ_score_test_tmp)
    concate  = pd.concat([threeD_file_test_tmp_mean, threeD_file_test_tmp_std], axis=1)
    #au_file_test.append(concate)
    threeD_file_test.append(threeD_file_test_tmp[['confidence', 'success', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'Y0', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10', 'Y11', 'Y12', 'Y13', 'Y14', 'Y15', 'Y16', 'Y17', 'Y18', 'Y19', 'Y20', 'Y21', 'Y22', 'Y23', 'Y24', 'Y25', 'Y26', 'Y27', 'Y28', 'Y29', 'Y30', 'Y31', 'Y32', 'Y33', 'Y34', 'Y35', 'Y36', 'Y37', 'Y38', 'Y39', 'Y40', 'Y41', 'Y42', 'Y43', 'Y44', 'Y45', 'Y46', 'Y47', 'Y48', 'Y49', 'Y50', 'Y51', 'Y52', 'Y53', 'Y54', 'Y55', 'Y56', 'Y57', 'Y58', 'Y59', 'Y60', 'Y61', 'Y62', 'Y63', 'Y64', 'Y65', 'Y66', 'Y67', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14', 'Z15', 'Z16', 'Z17', 'Z18', 'Z19', 'Z20', 'Z21', 'Z22', 'Z23', 'Z24', 'Z25', 'Z26', 'Z27', 'Z28', 'Z29', 'Z30', 'Z31', 'Z32', 'Z33', 'Z34', 'Z35', 'Z36', 'Z37', 'Z38', 'Z39', 'Z40', 'Z41', 'Z42', 'Z43', 'Z44', 'Z45', 'Z46', 'Z47', 'Z48', 'Z49', 'Z50', 'Z51', 'Z52', 'Z53', 'Z54', 'Z55', 'Z56', 'Z57', 'Z58', 'Z59', 'Z60', 'Z61', 'Z62', 'Z63', 'Z64', 'Z65', 'Z66', 'Z67']])                                                          
    # print("train_u:" , train_u) 
    # print("au_file:" , au_file)                                                             

print("---done loading 3D features Files----")
PHQ_score_array = np.array(PHQ_score)
PHQ_score_R = np.reshape(PHQ_score_array,(-1,1))
PHQ_score_test_array = np.array(PHQ_score_test)
PHQ_score_test_R = np.reshape(PHQ_score_test_array,(-1,1))
threeD_file = pd.concat(threeD_file, axis=0)    
threeD_file_test = pd.concat(threeD_file_test, axis=0)    
threeD_indexes = np.array(threeD_indexes)
threeD_indexes_for_sum = threeD_indexes[1:]-threeD_indexes[:-1]
threeD_indexes_for_sum = np.reshape(threeD_indexes_for_sum, (-1,1))

#print("2D indexes for sum:", threeD_indexes_for_sum)
#print("2D indexes for sum shape:", threeD_indexes_for_sum.shape)
#print("2D indexes for sum shape -5:", threeD_indexes_for_sum[:-5].shape)


#test = np.array(test_PHQ8B)
#print("is au any na:", pd.isna(au_file).any())
#print("is au test any na:", pd.isna(au_file_test).any())
scaler = MinMaxScaler()  
#Normalize train features and apply on test
threeD_file_transformed = scaler.fit_transform(threeD_file)  
threeD_file_test_transformed = scaler.transform(threeD_file_test) 

#print("training RFC Model...")
#Training a predicting Randomn Forest classifier
#regr = RandomForestClassifier(max_depth=2, random_state=0)              
#trainregr = regr.fit(threeD_file_transformed, PHQ_score_array) 
#joblib.dump(regr, "RFC Model 3D Points") 
#regr2 = joblib.load("RFC Model 3D Points")
#print("is 2d test any nan:", np.nan_to_num(threeD_file_test_transformed))
#print("is 2d test any inf:", np.nan_to_num(threeD_file_test_transformed))
#print("2d file test transformed:", threeD_file_test_transformed)



#predictionRegr = regr2.predict(np.nan_to_num(threeD_file_test_transformed))
#print("Prediction RFC ",predictionRegr)
#print("Prediction RFC shape",predictionRegr.shape)
#print("Prediction 3D Features indexes shape",threeD_indexes[:-1].shape)
#print("Prediction 3D Features shape", threeD_file_test_transformed[:-1].shape)

#predictionRegrSum = np.add.reduceat(predictionRegr,threeD_indexes[:-1],0).reshape(-1,1)
#print("Prediction RFC Sum shape",predictionRegrSum.shape)
#predictionRegrAVG = predictionRegrSum/threeD_indexes_for_sum
#predictionRegr_3BT = est.transform(predictionRegrAVG)
#predictionRegr_2BT = est2.transform(predictionRegrAVG)
#predictprobRFC = regr2.predict_proba(np.nan_to_num(threeD_file_test))
#print("---RFC Training and Prediction Done :)---")

#print("############3D Features Metrics############", file=open('output.txt', 'a'))
#print("############RFC Metrics############", file=open('output.txt', 'a'))

#rfcauc2instances = np.add.reduceat(predictprobRFC, [ 0., 10., 20.][:-1],1)
#rfcauc2sum = np.add.reduceat(rfcauc2instances,threeD_indexes[:-1],0)
#rfcauc2 = rfcauc2sum/threeD_indexes_for_sum
#rfcauc2 = rfcauc2[:,1]
#rfcauc3instances = np.add.reduceat(predictprobRFC, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
#rfcauc3sum = np.add.reduceat(rfcauc3instances,threeD_indexes[:-1],0)
#rfcauc3 = rfcauc3sum/threeD_indexes_for_sum
#print("rfcauc3:", rfcauc3)
#print("rfcauc2:", rfcauc2)

#regrROC3rfc = roc_auc_score(groundtruth, rfcauc3, multi_class='ovr')
#regrROC2rfc = roc_auc_score(groundtruth2, rfcauc2)
#accuracy3regr = accuracy_score(groundtruth, predictionRegr_3BT)
#accuracy2regr = accuracy_score(groundtruth2, predictionRegr_2BT)
#f13regr = f1_score(groundtruth, predictionRegr_3BT, average='macro')
#f12regr = f1_score(groundtruth2, predictionRegr_2BT)
#confusionmatrix3bins = confusion_matrix(groundtruth, predictionRegr_3BT).ravel()
#tn2regr, fp2regr, fn2regr, tp2regr = confusion_matrix(groundtruth2, predictionRegr_2BT).ravel()
#print("ROC AUC(3bins):", regrROC3rfc, file=open('output.txt', 'a'))
#print("ROC AUC(2bins):", regrROC2rfc, file=open('output.txt', 'a'))
#print("Accuracy (3bins):", accuracy3regr, file=open('output.txt', 'a'))
#print("Accuracy (2bins):", accuracy2regr, file=open('output.txt', 'a'))
#print("F1-Score (3bins):", f13regr, file=open('output.txt', 'a'))
#print("F1-Score (2bins):", f12regr, file=open('output.txt', 'a'))
#print("Confusion Matrix (3bins):", confusionmatrix3bins, file=open('output.txt', 'a'))
#rint("True Positive (2bins):", tp2regr, file=open('output.txt', 'a'))
#print("Flase Positive (2bins):", fp2regr, file=open('output.txt', 'a'))
#print("True Negative (2bins):", tn2regr, file=open('output.txt', 'a'))
#print("False Negative (2bins):", fn2regr, file=open('output.txt', 'a'))



#print("XGBoost Training and Prediction...")
# Training and Predicting XGBoost
#xgb_model = xgb.XGBClassifier(tree_method = 'gpu_hist')
#trainxgb = xgb_model.fit(threeD_file_transformed, PHQ_score_array)
#joblib.dump(xgb_model, "XGB model 3D") 
#xgb_model2 = joblib.load("XGB Model 2D")
#print("XGB model saved")

#predictionXGB = xgb_model.predict(np.nan_to_num(threeD_file_test_transformed))
#predictionXGBsum = np.add.reduceat(predictionXGB,threeD_indexes[:-1],0).reshape(-1,1)
#predictionXGBAVG = predictionXGBsum/threeD_indexes_for_sum
#predictionXGB_3BT = est.transform(predictionXGBAVG)
#predictionXGB_2BT = est2.transform(predictionXGBAVG)
#predictprobXGB = xgb_model.predict_proba(np.nan_to_num(threeD_file_test))
#print("---XGBoost Training and Prediction Done :)---")

# XGB metrics
#print("############XGBoost Metrics############", file=open('output.txt', 'a'))
#xgbauc2instances = np.add.reduceat(predictprobXGB, [ 0., 10., 20.][:-1],1)
#xgbauc2sum = np.add.reduceat(xgbauc2instances,threeD_indexes[:-1],0)
#xgbauc2 = xgbauc2sum/threeD_indexes_for_sum
#xgbauc2 = xgbauc2[:,1]
#xgbauc3instances = np.add.reduceat(predictprobXGB, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
#xgbauc3sum = np.add.reduceat(xgbauc3instances,threeD_indexes[:-1],0)
#xgbauc3 = xgbauc3sum/threeD_indexes_for_sum

#regrROC3xgb = roc_auc_score(groundtruth, xgbauc3,  multi_class='ovr')
#regrROC2xgb = roc_auc_score(groundtruth2, xgbauc2)
#accuracy3xgb = accuracy_score(groundtruth, predictionXGB_3BT)
#accuracy2xgb = accuracy_score(groundtruth2, predictionXGB_2BT)
#f13xgb = f1_score(groundtruth, predictionXGB_3BT, average='macro')
#f12xgb = f1_score(groundtruth2, predictionXGB_2BT)
#xgbconfusionmatrix3bins = confusion_matrix(groundtruth, predictionXGB_3BT).ravel()
#tn2xgb, fp2xgb, fn2xgb, tp2xgb = confusion_matrix(groundtruth2, predictionXGB_2BT).ravel()
#print("ROC AUC(3bins):", regrROC3xgb, file=open('output.txt', 'a'))
#print("ROC AUC(2bins):", regrROC2xgb, file=open('output.txt', 'a'))
#print("Accuracy (3bins):", accuracy3xgb, file=open('output.txt', 'a'))
#print("Accuracy (2bins):", accuracy2xgb, file=open('output.txt', 'a'))
#print("F1-Score (3bins):", f13xgb, file=open('output.txt', 'a'))
#print("F1-Score (2bins):", f12xgb, file=open('output.txt', 'a'))
#print("Confusion Matrix (3bins):", xgbconfusionmatrix3bins, file=open('output.txt', 'a'))
#print("True Positive (2bins):", tp2xgb, file=open('output.txt', 'a'))
#print("Flase Positive (2bins):", fp2xgb, file=open('output.txt', 'a'))
#print("True Negative (2bins):", tn2xgb, file=open('output.txt', 'a'))
#print("False Negative (2bins):", fn2xgb, file=open('output.txt', 'a'))


#print("Naive Bayes Training and Prediction...")
#gnb = GaussianNB()
#traingnb = gnb.fit(threeD_file_transformed, PHQ_score_array)
#joblib.dump(gnb, "GNB model 3D") 
##gnb2 = joblib.load("GNB Model 2D")
#print("GNB model saved")
#predictionGNB = gnb.predict(np.nan_to_num(threeD_file_test_transformed))
#predictionGNBSum = np.add.reduceat(predictionGNB,threeD_indexes[:-1],0).reshape(-1,1)
#predictionGNBAVG = predictionGNBSum/threeD_indexes_for_sum
#predictionGNB_3BT = est.transform(predictionGNBAVG)
#predictionGNB_2BT = est2.transform(predictionGNBAVG)
#predictprobGNB = gnb.predict_proba(np.nan_to_num(threeD_file_test))

#gnbauc2instances = np.add.reduceat(predictprobGNB, [ 0., 10., 20.][:-1],1)
#gnbauc2sum = np.add.reduceat(gnbauc2instances,threeD_indexes[:-1],0)
#gnbauc2 = gnbauc2sum/threeD_indexes_for_sum
#gnbauc2 = gnbauc2[:,1]
#gnbauc3instances = np.add.reduceat(predictprobGNB, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
#gnbauc3sum = np.add.reduceat(gnbauc3instances,threeD_indexes[:-1],0)
#gnbauc3 = gnbauc3sum/threeD_indexes_for_sum
#print("---Naive Bayes Training and Prediction Done :)---")

# metrics for GNB
#print("############Naive Bayes Metrics############", file=open('output.txt', 'a'))
#regrROC3gnb = roc_auc_score(groundtruth, gnbauc3,  multi_class='ovr')
#regrROC2gnb = roc_auc_score(groundtruth2, gnbauc2)
#accuracygnb = accuracy_score(groundtruth, predictionGNB_3BT)
#accuracy2gnb = accuracy_score(groundtruth2, predictionGNB_2BT)
#f13gnb = f1_score(groundtruth, predictionGNB_3BT, average='macro')
#f12gnb = f1_score(groundtruth2, predictionGNB_2BT)
#gnbconfusionmatrix3bins = confusion_matrix(groundtruth, predictionGNB_3BT).ravel()
#tn2gnb, fp2gnb, fn2gnb, tp2gnb = confusion_matrix(groundtruth2, predictionGNB_2BT).ravel()
#print("ROC AUC(3bins):", regrROC3gnb, file=open('output.txt', 'a'))
#print("ROC AUC(2bins):", regrROC2gnb, file=open('output.txt', 'a'))
#print("Accuracy (3bins):", accuracygnb, file=open('output.txt', 'a'))
#print("Accuracy (2bins):", accuracy2gnb, file=open('output.txt', 'a'))
#print("F1-Score (3bins):", f13gnb, file=open('output.txt', 'a'))
#print("F1-Score (2bins):", f12gnb, file=open('output.txt', 'a'))
#print("Confusion Matrix (3bins):", gnbconfusionmatrix3bins, file=open('output.txt', 'a'))
#print("True Positive (2bins):", tp2gnb, file=open('output.txt', 'a'))
#print("Flase Positive (2bins):", fp2gnb, file=open('output.txt', 'a'))
#print("True Negative (2bins):", tn2gnb, file=open('output.txt', 'a'))
#print("False Negative (2bins):", fn2gnb, file=open('output.txt', 'a'))

print("SVM Training and Prediction...")

clf=svm.LinearSVC()
trainclf = clf.fit(threeD_file_transformed, PHQ_score_array)
joblib.dump(clf, "SVM Model 3D") 
#clf2 = joblib.load("SVM Model 2D")
print("SVM model saved")
predictionSVM = clf.predict(np.nan_to_num(threeD_file_test_transformed))  
predictionSVMSUM =  np.add.reduceat(predictionSVM,threeD_indexes[:-1],0).reshape(-1,1)
predictionSVMAVG = predictionSVMSUM/threeD_indexes_for_sum
predictionSVM_3BT = est.transform(predictionSVMAVG)
predictionSVM_2BT = est2.transform(predictionSVMAVG)
predictprobSVM = clf._predict_proba_lr(np.nan_to_num(threeD_file_test))
print("---SVM Training and Prediction Done :)---")

print("############3D Features SVM Metrics############", file=open('output.txt', 'a'))
svmauc2instances = np.add.reduceat(predictprobSVM, [ 0., 10., 20.][:-1],1)
svmauc2sum = np.add.reduceat(svmauc2instances, threeD_indexes[:-1],0)
svmauc2 = svmauc2sum/threeD_indexes_for_sum
svmauc2 = svmauc2[:,1]
svmauc3instances = np.add.reduceat(predictprobSVM, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
svmauc3sum = np.add.reduceat(svmauc3instances,threeD_indexes[:-1],0)
svmauc3 = svmauc3sum/threeD_indexes_for_sum

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