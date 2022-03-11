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
import xgboost as xgb                         
import joblib  
import zipfile
import pickle
import sys
import gc
import time

def timing(func):
    def wrapper(args):
        print("Running function {}".format(func.__name__))
        t1 = time.time()
        res = func(*args)
        t2 = time.time()
        period = t2 - t1
        print("{} took {} hour {} min {} sec".format(func.__name__, period // 3600, (period % 3600) // 60,
                                                     int(period % 60)))
        print("")
        return res

    return wrapper

@timing
def gen_param(model_name):
    if model_name =='RFC':
        parameters = {
            'n_estimators': [50, 100, 200], 
            'criterion': ["gini", "entropy"], 
            'max_depth': [5], 
            'min_samples_split': [2],
            'class_weight': ["balanced"], 
            'random_state': [1],
            }
        clf = RandomForestClassifier()

    elif model_name =='LR':
        parameters = {
            'loss':['log'], 
            'penalty':['l2'], 
            'max_iter':[10000], 
            'tol':[1e-4],
            'class_weight': ["balanced"], 
            'random_state': [1],
            'early_stopping': [True],
            }
        clf= SGDClassifier()

    elif model_name == 'XGBOOST':
        parameters = {
            'n_estimators': [50, 100, 200],
            'learning_rate':[0.0001, 0.001, 0.01], 
            # 'gamma':[5], 
            'use_label_encoder':[False],
            'max_depth':[5],
            'min_child_weight':[1], 
            # 'subsample':[0.7], 
            # 'colsample_bytree':[0.6],
            'random_state': [1],
            'gpu_id':[0],
            'tree_method':['gpu_hist']
            }
        clf = xgb.XGBClassifier()

    elif model_name =='NB':
        parameters = {
            'var_smoothing':[1e-9],
            }
        clf = GaussianNB()

    return model_name, parameters, clf

@timing
def train(model_name, parameters, clf, au_file_transformed, PHQ_score_array):
    # grid search and get best parameters
    model = GridSearchCV(clf, param_grid=parameters, scoring='f1_macro', n_jobs=3, refit=True, cv=5)
    model.fit(au_file_transformed, PHQ_score_array)
    gc.collect()
    
    print("Best Hyperparameter", model.best_params_)

    return model_name, model

@timing
def infer(model_name, model, au_file_test_transformed):
    # prepare model directory
    output_path = './{}/'.format(model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    joblib.dump(model, output_path+"{}_model".format(model_name)) 

    # predict on the new model and hold out test data
    pred = model.predict(np.nan_to_num(au_file_test_transformed))
    pred_prob = model.predict_proba(np.nan_to_num(au_file_test_transformed))
    gc.collect()

    # save predictions and the probabilities
    np.save(output_path+'{}_prediction.npy'.format(model_name), pred)
    np.save(output_path+'{}_pred_prob.npy'.format(model_name), pred_prob)

    return pred, pred_prob

@timing
def _eval(pred, pred_prob, groundtruth, groundtruth2, est, est2, AU_indexes, AU_indexes_for_sum):
    predictionSum = np.add.reduceat(pred,AU_indexes[:-1],0).reshape(-1,1)
    predictionAVG = predictionSum/AU_indexes_for_sum
    prediction_3BT = est.transform(predictionAVG)
    prediction_2BT = est2.transform(predictionAVG)
    gc.collect()

    rfcauc2instances = np.add.reduceat(pred_prob, [ 0., 10., 20.][:-1],1)
    rfcauc2sum = np.add.reduceat(rfcauc2instances,AU_indexes[:-1],0)
    rfcauc2 = rfcauc2sum/AU_indexes_for_sum
    rfcauc2 = rfcauc2[:,1]
    rfcauc3instances = np.add.reduceat(pred_prob, [ 0.,  6.66666667, 13.33333333, 20.][:-1],1)
    rfcauc3sum = np.add.reduceat(rfcauc3instances,AU_indexes[:-1],0)
    rfcauc3 = rfcauc3sum/AU_indexes_for_sum
    gc.collect()

    regrROC3rfc = roc_auc_score(groundtruth, rfcauc3, multi_class='ovr')
    regrROC2rfc = roc_auc_score(groundtruth2, rfcauc2)
    gc.collect()

    accuracy3regr = accuracy_score(groundtruth, prediction_3BT)
    accuracy2regr = accuracy_score(groundtruth2, prediction_2BT)
    gc.collect()

    f13regr = f1_score(groundtruth, prediction_3BT, average='macro')
    f12regr = f1_score(groundtruth2, prediction_2BT)
    gc.collect()

    confusionmatrix3bins = confusion_matrix(groundtruth, prediction_3BT).ravel()
    confusionmatrix2bins = confusion_matrix(groundtruth2, prediction_2BT).ravel()
    gc.collect()

    print("ROC AUC(3bins):", round(regrROC3rfc,4))
    print("ROC AUC(2bins):", round(regrROC2rfc,4))
    print("Accuracy (3bins):", round(accuracy3regr,4))
    print("Accuracy (2bins):", round(accuracy2regr,4))
    print("F1-Score (3bins):", round(f13regr,4))
    print("F1-Score (2bins):", round(f12regr,4))
    print("Confusion Matrix (3bins):", confusionmatrix3bins)
    print("True Positive (2bins):", confusionmatrix2bins)
    return

@timing
def _eval2Bin(pred, pred_prob, groundtruth2, AU_indexes, AU_indexes_for_sum):
    predictionSum = np.add.reduceat(pred,AU_indexes[:-1],0).reshape(-1,1)
    predictionAVG = predictionSum/AU_indexes_for_sum
    prediction_2BT = (predictionAVG>0.5).astype(int)
    gc.collect()

    rfcauc2sum = np.add.reduceat(pred_prob,AU_indexes[:-1],0)
    rfcauc2 = rfcauc2sum/AU_indexes_for_sum
    rfcauc2 = rfcauc2[:,1]

    gc.collect()
    regrROC2rfc = roc_auc_score(groundtruth2, rfcauc2)
    gc.collect()

    accuracy2regr = accuracy_score(groundtruth2, prediction_2BT)
    gc.collect()

    f12regr = f1_score(groundtruth2, prediction_2BT)
    gc.collect()

    confusionmatrix2bins = confusion_matrix(groundtruth2, prediction_2BT).ravel()
    gc.collect()

    print("ROC AUC(2bins):", round(regrROC2rfc,4))
    print("Accuracy (2bins):", round(accuracy2regr,4))
    print("F1-Score (2bins):", round(f12regr,4))
    print("True Positive (2bins):", confusionmatrix2bins)
    return


@timing
def readData(outputpath):
    au_file_transformed = np.load(outputpath+'AUDIO_train.npy')
    au_file_test_transformed= np.load(outputpath+'AUDIO_test.npy')
    gc.collect()

    PHQ_score_array_2Bin = np.load(outputpath+'AUDIO_PHQ_2Bin_train.npy').ravel()
    # PHQ_score_array = np.load(outputpath+'AUDIO_PHQ_train.npy')
    # PHQ_score_test_array = np.load(outputpath+'AUDIO_PHQ_test.npy')
    gc.collect()

    AU_indexes = np.load(outputpath+'AUDIO_indexes.npy')
    AU_indexes_for_sum = np.load(outputpath+'AUDIO_indexes_sum.npy')
    gc.collect()

    groundtruth2 = np.load(outputpath+'groundtruth2bin.npy')
    # groundtruth = np.load(outputpath+'groundtruth3bin.npy')
    gc.collect()

    with open(outputpath+'est2.pkl','rb') as f:
        est2 = pickle.load(f)
    with open(outputpath+'est3.pkl','rb') as f:
        est = pickle.load(f)
    gc.collect()

    train_data = [au_file_transformed, PHQ_score_array_2Bin]
    infer_data = [au_file_test_transformed]
    eval_data = [ groundtruth2, AU_indexes, AU_indexes_for_sum]

    return train_data, infer_data, eval_data

if __name__=='__main__':
    path = '../data/'
    # 
    models = ['NB','RFC','XGBOOST','LR']

    train_data, infer_data, eval_data = readData([path])

    for model_name in models:
        print("===========================================================")
        print("Start for {}".format(model_name))
        best_s = gen_param([model_name])
        best_m = train([*best_s, *train_data])
        best_p = infer([*best_m, *infer_data])
        best_e = _eval2Bin([*best_p, *eval_data])















                                                                                
 