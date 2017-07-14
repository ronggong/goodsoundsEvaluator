from keras.models import load_model
import numpy as np
import pickle
import os
import pandas as pd
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def loadData(feature_string):
    # feature and label preparation
    filename_testData     = './dataset/bag-of-feature-framelevel/testData_'+feature_string+'.pkl'

    with open(filename_testData, 'rb') as f:
        X_test, Y_test  = pickle.load(f)

    return X_test, Y_test

def getTestResults(observations, path_keras_cnn):
    """
    Load CNN model to calculate test results
    :param observations:
    :return:
    """

    model = load_model(path_keras_cnn)

    testResults = model.predict(observations, batch_size=16, verbose=0)

    return testResults


def metrics(y_test, y_pred):
    print classification_report(y_test, y_pred)
    # print confusion_matrix(y_test, y_pred)
    print "Accuracy:"
    print accuracy_score(y_test, y_pred)
    print "Micro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='micro')
    print "Macro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='macro')

def testProcess(filename_test_set, filename_keras_model):
    with open(filename_test_set, 'rb') as f:
        X_test, Y_true = pickle.load(f)

    max_length = 1401

    for ii in xrange(len(X_test)):
        X_test[ii] = sequence.pad_sequences(X_test[ii].transpose(), maxlen=max_length, dtype='float32')

    X_test = np.array(X_test, dtype='float32')

    testResults = getTestResults(X_test, filename_keras_model)

    Y_pred = [0 if tr > 0.5 else 1 for tr in testResults[:,0]]

    # num_wrong_pred = sum([abs(Y_test[ii] - Y_pred[ii]) for ii in xrange(len(Y_test))])

    metrics(Y_true, Y_pred)

def featureSegmentation(feature, seg=50):
    """
    segment feature into 50 frames segment
    :param feature:
    :param seg:
    :return:
    """

    length = feature.shape[1]
    list_feature_out = []
    ii = 0
    while length>seg:
        list_feature_out.append(feature[:,ii*seg:(ii+1)*seg])
        length -= seg
        ii += 1

    # last segment
    if length>seg*0.2:
        last_seg = feature[:,ii*seg:]
        for jj in range(int(seg/length)+1):
            last_seg = np.hstack([last_seg, last_seg])
        last_seg = last_seg[:,:seg]
        list_feature_out.append(last_seg)
    return list_feature_out

def testProcessAugmentation(filename_test_set, filename_keras_model):
    with open(filename_test_set, 'rb') as f:
        X_test, Y_true = pickle.load(f)

    model = load_model(filename_keras_model)

    Y_pred = []
    for ii in xrange(len(Y_true)):
        X_test[ii] = featureSegmentation(X_test[ii].transpose())
        X_test[ii] = np.array(X_test[ii], dtype='float32')
        pred = model.predict(X_test[ii], verbose=0)
        print pred
        Y_pred.append(np.mean(pred))

    Y_pred = [0 if tr > 0.5 else 1 for tr in Y_pred]

    metrics(Y_true, Y_pred)


def testProcessVariableLength(feature_test, label_test, filename_keras_model):

    model = load_model(filename_keras_model)

    Y_pred = []
    for x in feature_test:
        x = np.expand_dims(x, axis=0)
        testResults = model.predict(x)
        print testResults
        Y_pred.append(testResults[0][0])

    Y_pred = [1 if tr > 0.5 else 0 for tr in Y_pred]

    metrics(label_test, Y_pred)


if __name__ == '__main__':
    # filename_test_set = './dataset/testData_timbre.pkl'
    # filename_keras_model = './models/crnn_model_cw_timbre_gru1layer_32nodes_thin_vgg_65k_bidirectional.h5'
    # testProcess(filename_test_set, filename_keras_model)
    #
    # filename_test_set       = './dataset/testData_pitch.pkl'
    # filename_keras_model    = './models/crnn_model_cw_pitch_gru1layer_32nodes_fat_vgg.h5'
    # testProcess(filename_test_set, filename_keras_model)

    filename_test_set       = './dataset/testData_dynamics.pkl'
    filename_keras_model    = './models/crnn_model_cw_dynamics_gru1layer_32nodes_thin_vgg.h5'
    testProcess(filename_test_set, filename_keras_model)

    # filename_test_set       = './dataset/testData_richness.pkl'
    # filename_keras_model    = './models/crnn_model_cw_richness_gru1layer_32nodes_thin_vgg.h5'
    # testProcess(filename_test_set, filename_keras_model)

    filename_test_set       = './dataset/testData_attack.pkl'
    filename_keras_model    = './models/crnn_model_cw_attack_gru1layer_32nodes_fat_vgg.h5'
    testProcess(filename_test_set, filename_keras_model)

    # feature_test, label_test = loadData('timbre')
    # testProcessVariableLength(feature_test, label_test, './test_model_2layers.h5')

    # filename_test_set = './dataset/testData_timbre.pkl'
    # filename_keras_model = './models/augmentation/crnn_model_cw_timbre_gru1layer_32nodes_fat_vgg_65k_augmentation.h5'
    # testProcessAugmentation(filename_test_set, filename_keras_model)