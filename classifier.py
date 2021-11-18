from _typeshed import OpenTextMode
import sklearn
import os
import pandas as pd  
import numpy as np  
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import MinMaxScaler                         
  

au = pd.read_csv('/Users/jessieeteng/Downloads/300_P/300_COVAREP.csv ', sep='\t')



au = pd.read_csv('/Users/gggwen/Downloads/300_P/300_CLNF_AUs.txt')      
train_list = pd.read_csv('/Users/gggwen/Downloads/train_split_Depression...: _AVEC2017.csv')                                                         
train_user_id = train_list['Participant_ID']                           
train_PHQ8B = train_list['PHQ8_Binary']                                

for train_u in train_user_id:    
    au_file = pd.read_csv('/Users/gggwen/Downloads/{}_P/{}_clnp_aus.txt...: '.format(train_u,train_u))                                                                   

au_file = np.random.rand(100,13)                                       

regr = RandomForestRegressor(max_depth=2, random_state=0)              

regr.fit(au_file, train_PHQ8B)                                         

scaler = MinMaxScaler()                                                

au_file = scaler.fit_transform(au_file)                                

au_file_test = scaler.tranform(au_file_test)                           
regr.fit(au_file, train_PHQ8B)                                         

prediction = regr.predict(au_file_test)                                

au_file = [] 
for train_u in train_user_id: 
  au_file_tmp = pd.read_csv('/Users/gggwen/Downloads/{}_P/{}_clnp_aus.txt'.format(train_u,train_u)) 
    au_file.append(au_File_tmp) 
      pd.concat(au_file, axis=0) 




                                                                                
 