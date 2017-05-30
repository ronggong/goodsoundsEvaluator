from keras.models import load_model
import hpcDLScripts.crnn_evaluator_gru1layer_32nodes_thin_vgg_35k as crnn_evaluator
import hpcDLScripts.crnn_evaluator_nogru as crnn_evaluator_nogru
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import pickle
from filePath import *


def extractWeights(filename_keras_model, filename_keras_weights):
    model = load_model(join(dir_path,filename_keras_model))
    model.save_weights(join(dir_path,filename_keras_weights))

def processSaveWeights():

    # filename_keras_model = './models/crnn_model_cw_timbre_gru1layer_10nodes_model12.h5'
    # filename_keras_weights = './weights/crnn_model_cw_timbre_gru1layer_10nodes_model12_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    # #
    filename_keras_model    = './models/crnn_model_cw_pitch_gru1layer_10nodes_model15.h5'
    filename_keras_weights = './weights/crnn_model_cw_pitch_gru1layer_10nodes_model15_weights.h5'
    extractWeights(filename_keras_model, filename_keras_weights)
    # #
    # filename_keras_model    = './models/crnn_model_cw_dynamics_gru1layer_10nodes_model16.h5'
    # filename_keras_weights = './weights/crnn_model_cw_dynamics_gru1layer_10nodes_model16_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    # #
    # filename_keras_model    = './models/crnn_model_cw_richness_gru1layer_10nodes_model17.h5'
    # filename_keras_weights = './weights/crnn_model_cw_richness_gru1layer_10nodes_model17_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    # #
    # filename_keras_model    = './models/crnn_model_cw_attack_gru1layer_10nodes_model18.h5'
    # filename_keras_weights = './weights/crnn_model_cw_attack_gru1layer_10nodes_model18_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    #
    #
    #
    # filename_keras_model = './models/crnn_model_cw_timbre_nogru_10k_model11.h5'
    # filename_keras_weights = './weights/crnn_model_cw_timbre_nogru_10k_model11_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    # #
    # filename_keras_model    = './models/crnn_model_cw_pitch_nogru_model19.h5'
    # filename_keras_weights = './weights/crnn_model_cw_pitch_nogru_model19_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    # #
    # filename_keras_model    = './models/crnn_model_cw_dynamics_nogru_model20.h5'
    # filename_keras_weights = './weights/crnn_model_cw_dynamics_nogru_model20_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    # #
    # filename_keras_model    = './models/crnn_model_cw_richness_nogru_model21.h5'
    # filename_keras_weights = './weights/crnn_model_cw_richness_nogru_model21_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    # #
    # filename_keras_model    = './models/crnn_model_cw_attack_nogru_model22.h5'
    # filename_keras_weights = './weights/crnn_model_cw_attack_nogru_model22_weights.h5'
    # extractWeights(filename_keras_model, filename_keras_weights)
    #

def testData(filename_test_set):

    with open(join(dir_path,filename_test_set), 'rb') as f:
        X_test, Y_true = pickle.load(f)

    max_length = 1401

    for ii in xrange(len(X_test)):
        X_test[ii] = sequence.pad_sequences(X_test[ii].transpose(), maxlen=max_length, dtype='float32')

    X_test = np.array(X_test, dtype='float32')

    return X_test, Y_true

def calculateResults(X, filename_keras_weights):
    # crnn_model = crnn_evaluator.crnn_model(1, 3, 3, 1401, 0.3, include_top=False)
    crnn_model = crnn_evaluator_nogru.crnn_model(1, 3, 3, 1401, 0.3, include_top=False)

    crnn_model.load_weights(join(dir_path,filename_keras_weights), by_name=True)

    testResults = crnn_model.predict(X, batch_size=16, verbose=1)

    return testResults


def trainData(feature_string):
    filename_all_set, filename_scaler, filename_trainIndex = crnn_evaluator.loadDataPath(feature_string)

    X_train, X_val, X_train_validation, Y_train, Y_val, Y_train_validation_categorical, Y_train_validation, class_weights, max_length = \
        crnn_evaluator.loadData(feature_string, filename_all_set, filename_scaler, filename_trainIndex)

    return X_train_validation, Y_train_validation

def saveFeatureCsv(X, y, filename_keras_weights, feature_string, train_test_string):

    feature = calculateResults(X, filename_keras_weights)

    df = pd.DataFrame(feature)

    df['class'] = pd.Series(y, index=df.index)

    df.to_csv(join(dir_path,'./dataset/transfer_feature/'+train_test_string+'Data_'+feature_string+'_nogru.csv'))

def saveFeatureTrainTest(feature_string, filename_keras_weights, filename_test_set):
    X_train,y_train = trainData(feature_string)
    saveFeatureCsv(X_train, y_train, filename_keras_weights, feature_string, 'train')
    X_test, y_test = testData(filename_test_set)
    saveFeatureCsv(X_test, y_test, filename_keras_weights, feature_string, 'test')

max_length = 1401

# processSaveWeights()

# filename_test_set = './dataset/testData_timbre.pkl'
# filename_keras_weights = './weights/crnn_model_cw_timbre_gru1layer_10nodes_model12_weights.h5'
# feature_string = 'timbre'
# saveFeatureTrainTest(feature_string, filename_keras_weights, filename_test_set)

filename_test_set       = './dataset/testData_pitch.pkl'
# filename_keras_weights = './weights/crnn_model_cw_pitch_gru1layer_10nodes_model15_weights.h5'
filename_keras_weights = './weights/crnn_model_cw_pitch_nogru_model19_weights.h5'

feature_string = 'pitch'
saveFeatureTrainTest(feature_string, filename_keras_weights, filename_test_set)

#
# filename_test_set       = './dataset/testData_dynamics.pkl'
# filename_keras_weights = './weights/crnn_model_cw_dynamics_gru1layer_10nodes_model16_weights.h5'
# feature_string = 'dynamics'
# saveFeatureTrainTest(feature_string, filename_keras_weights, filename_test_set)
#
#
# filename_test_set       = './dataset/testData_richness.pkl'
# filename_keras_weights = './weights/crnn_model_cw_richness_gru1layer_10nodes_model17_weights.h5'
# feature_string = 'richness'
# saveFeatureTrainTest(feature_string, filename_keras_weights, filename_test_set)
#
#
# filename_test_set       = './dataset/testData_attack.pkl'
# filename_keras_weights = './weights/crnn_model_cw_attack_gru1layer_10nodes_model18_weights.h5'
# feature_string = 'attack'
# saveFeatureTrainTest(feature_string, filename_keras_weights, filename_test_set)