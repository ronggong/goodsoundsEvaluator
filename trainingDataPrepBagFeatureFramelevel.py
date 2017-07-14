import pickle
import pandas as pd
import os
from filePath import path_dataset
import numpy as np

def loadDataPath():

    path_feature = os.path.join(path_dataset, 'feature_framelevel_standarized')
    filename_timbre_label_train = os.path.join(path_dataset, 'feature_framelevel', 'y_train_timbre.csv')
    filename_timbre_label_test = os.path.join(path_dataset, 'feature_framelevel', 'y_test_timbre.csv')


    return path_feature, filename_timbre_label_train, filename_timbre_label_test

def loadDataVariablelength(path_feature, filename_timbre_label_train, filename_timbre_label_test):

    label_train = pd.DataFrame.from_csv(filename_timbre_label_train,header=None)
    label_test = pd.DataFrame.from_csv(filename_timbre_label_test,header=None)
    filename_train = label_train.index
    filename_test = label_test.index
    label_train = np.transpose(label_train.values)[0]
    label_test = np.transpose(label_test.values)[0]

    feature_train = []
    feature_test = []
    for ii, fn_train in enumerate(filename_train):
        print('loading train ', ii)
        feature_train.append(pd.DataFrame.from_csv(os.path.join(path_feature,fn_train+'.csv')).values)
    for ii, fn_test in enumerate(filename_test):
        print('loading test ', ii)
        feature_test.append(pd.DataFrame.from_csv(os.path.join(path_feature,fn_test+'.csv')).values)
    return feature_train, label_train, feature_test, label_test

if __name__ == '__main__':
    path_feature, filename_timbre_label_train, filename_timbre_label_test = loadDataPath()
    feature_train, label_train, feature_test, label_test = loadDataVariablelength(path_feature,
                                                                                  filename_timbre_label_train,
                                                                                  filename_timbre_label_test)
    pickle.dump((feature_train, label_train), open("./dataset/bag-of-feature-framelevel/trainData_timbre.pkl","wb"))
    pickle.dump((feature_test, label_test), open("./dataset/bag-of-feature-framelevel/testData_timbre.pkl", "wb"))