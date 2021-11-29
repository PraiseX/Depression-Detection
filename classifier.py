import sklearn
import os
import pandas as pd  
import glob
import numpy as np  
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn import svm


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

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')


test_list = pd.read_csv('D:\MQP\dataset\DAICWOZ\dev_split_Depression_AVEC2017.csv')                                                         
test_user_id = test_list['Participant_ID']                           
test_PHQ8B = test_list['PHQ8_Score'] 


au_file = [] 
au_file_test = []

#make sure all the participants from the training split are in folder
for train_u in train_user_id: 
    #depending on the file use different seperators
    au_file_tmp = pd.read_csv(path+'{}_CLNF_AUs.txt'.format(train_u), sep = ', ') 
    au_file.append(au_file_tmp[['confidence', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']])  
    label = np.array(['confidence', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c'])
    cols=pd.MultiIndex.from_product(label)
    au_reshaped = np.reshape(au_file,(29565, 22))
    au_frame = pd.DataFrame(au_reshaped)
    print("au_file:" , au_frame.mean(axis=1))

for test_u in test_user_id: 
    #depending on the file use different seperators
    au_file_test_tmp = pd.read_csv(path+'{}_CLNF_AUs.txt'.format(train_u), sep = ', ') 
    au_file_test.append(au_file_test_tmp[['confidence', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c', 'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c']]) 
    
    # print("train_u:" , train_u) 
    # print("au_file:" , au_file)                                                             

# avg_score = train_PHQ8B.mean / train_PHQ8B.__len__ - 1                      

au_file = pd.concat(au_file, axis=0)    


scaler = MinMaxScaler()                                                

au_file = scaler.fit_transform(au_file)  

regr = RandomForestRegressor(max_depth=2, random_state=0)              
clf=svm.SVC()
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)

# groundtruth = train_list['PHQ8_Score'] 
                    

au_file_test = scaler.transform(au_file_test) 



truerger = regr.fit(au_file, train_PHQ8B) 
clf.fit(au_file, train_PHQ8B)
xgb_model.fit(au_file, train_PHQ8B)


predictionRegr = regr.predict(au_file_test)   
predictionSVM = clf.predict(au_file_test)                             
predictionXGB = xgb_model.predict(au_file_test)

est.fit(train_PHQ8B)      

threebins = est.transform(predictionRegr)
threebinsgroundtruth = est.transform(test_PHQ8B)

#average all predictions
#

regrROC = sklearn.metrics.roc_auc_score(test_PHQ8B, clf.predict_proba(au_file_test))




 




                                                                                
 