
# https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py

import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.preprocessing import sequence
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, Merge, ELU, GRU, Permute, Masking
from keras.regularizers import l2


def loadData(feature_string):
    # feature and label preparation
    # filename_all_set    = '../dataset/allData.pkl'
    # filename_scaler     = '../dataset/scaler_'+feature_string+'_train.pkl'
    # filename_trainIndex = '../dataset/trainIndex_'+feature_string+'.pkl'
    #
    # filename_train_validation_set = '/scratch/rgongcrnnSingleSoundEvaluator_'+feature_string+'_gru1layer_10nodes/data/allData.pkl'
    # filename_train_validation_set = '../dataset/trainData_timbre.pkl'

    filename_all_set    = '/scratch/rgongcrnnSingleSoundEvaluator_'+feature_string+'_gru1layer_10nodes_dense/data/allData.pkl'
    filename_scaler     = '/scratch/rgongcrnnSingleSoundEvaluator_'+feature_string+'_gru1layer_10nodes_dense/data/scaler_' + feature_string + '_train.pkl'
    filename_trainIndex = '/scratch/rgongcrnnSingleSoundEvaluator_'+feature_string+'_gru1layer_10nodes_dense/data/trainIndex_'+feature_string+'.pkl'

    with open(filename_all_set, 'rb') as f:
        X_all, Y_all  = pickle.load(f)

    with open(filename_trainIndex, 'rb') as f:
        trainIndex  = pickle.load(f)

    with open(filename_scaler, 'rb') as f:
        scaler  = pickle.load(f)

    # dimension order: [timbre, pitch, dynamics, richness, attack]

    if feature_string == 'timbre':
        label_all = [label_single_sound[0] for label_single_sound in Y_all]
    elif feature_string == 'pitch':
        label_all = [label_single_sound[1] for label_single_sound in Y_all]
    elif feature_string == 'dynamics':
        label_all = [label_single_sound[2] for label_single_sound in Y_all]
    elif feature_string == 'richness':
        label_all = [label_single_sound[3] for label_single_sound in Y_all]
    elif feature_string == 'attack':
        label_all = [label_single_sound[4] for label_single_sound in Y_all]

    X_train_validation = [scaler.transform(X_all[ti]) for ti in trainIndex]
    Y_train_validation = [label_all[ti] for ti in trainIndex]

    # with open(filename_train_validation_set, 'rb') as f:
    #     X_train_validation, Y_train_validation  = pickle.load(f)


    # print(X_train_validation[0].shape)

    max_length = 1401
    # for spectro in X_all:
    #     if spectro.shape[0] > max_length:
    #         max_length = spectro.shape[0]

    X_train, X_val, Y_train, Y_val = train_test_split(X_train_validation, Y_train_validation, test_size=0.2, stratify=Y_train_validation)

    for ii in xrange(len(X_train)):
        X_train[ii] = sequence.pad_sequences(X_train[ii].transpose(), maxlen=max_length, dtype='float32')
    for ii in xrange(len(X_val)):
        X_val[ii] = sequence.pad_sequences(X_val[ii].transpose(), maxlen=max_length, dtype='float32')
    for ii in xrange(len(X_train_validation)):
        X_train_validation[ii] = sequence.pad_sequences(X_train_validation[ii].transpose(), maxlen=max_length, dtype='float32')


    # print(X_train[0].shape)
    # print(X_val[0].shape)
    # print(len(X_train))

    X_train             = np.array(X_train, dtype='float32')
    X_val               = np.array(X_val, dtype='float32')
    X_train_validation  = np.array(X_train_validation, dtype='float32')

    # print(Y_train_validation)
    # print(type(Y_train_validation))

    class_weights = compute_class_weight('balanced',[0,1],Y_train_validation)
    print(class_weights)
    class_weights = {0:class_weights[0], 1:class_weights[1]}

    Y_train             = to_categorical(Y_train)
    Y_val               = to_categorical(Y_val)
    Y_train_validation  = to_categorical(Y_train_validation)

    print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

    return X_train, X_val, X_train_validation, Y_train, Y_val, Y_train_validation, class_weights, max_length


def crnn_model(filter_density, pool_n_row, pool_n_col, nlen, dropout, include_top=True):

    reshape_dim = (80, nlen, 1)
    input_dim = (80, nlen)

    model = Sequential()

    # conv 0
    model.add(Reshape(reshape_dim, input_shape=input_dim))
    model.add(Permute((2, 1, 3)))
    model.add(Convolution2D(int(10*filter_density), 3, 3, border_mode='valid',
                      input_shape=reshape_dim, dim_ordering='tf',
                      init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='tf'))
    model.add(Dropout(0.1))

    # conv 1
    model.add(Convolution2D(int(20*filter_density), 3, 3, border_mode='valid',
                            input_shape=reshape_dim, dim_ordering='tf',
                            init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', dim_ordering='tf'))
    model.add(Dropout(0.1))

    # conv 2
    model.add(Convolution2D(int(20 * filter_density), 3, 3, border_mode='valid',
                            input_shape=reshape_dim, dim_ordering='tf',
                            init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', dim_ordering='tf'))
    model.add(Dropout(0.1))

    # conv 3
    model.add(Convolution2D(int(20 * filter_density), 3, 3, border_mode='valid',
                            input_shape=reshape_dim, dim_ordering='tf',
                            init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(3, 1), border_mode='valid', dim_ordering='tf'))
    model.add(Dropout(0.1))

    print(model.output_shape)

    # model.output_shape[1]*model.output_shape[2]

    # model.add(Reshape((model.output_shape[1]*model.output_shape[2],20)))
    model.add(Flatten())

    model.add(Dense(output_dim=10, init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())
    model.add(Dropout(0.3))

    # model.add(Masking(mask_value=0.0))

    # GRU block 1, 2, output
    # model.add(GRU(10, return_sequences=True, name='gru1'))
    # model.add(GRU(10, return_sequences=False, name='gru2'))
    # model.add(Dropout(0.3))
    if include_top:
        model.add(Dense(2, activation='softmax', name='output'))

    return model

def train_model(filter_density,
                pool_n_row,
                pool_n_col,
                nlen,
                dropout,
                file_path_model,
                X_train,
                X_val,
                X_train_validation,
                Y_train,
                Y_val,
                Y_train_validation,
                class_weights):
    """
    train final model save to model path
    """

    model_merged_0 = crnn_model(filter_density, pool_n_row, pool_n_col, nlen, dropout)

    optimizer = Adam()

    model_merged_0.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model_merged_0.summary()

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    hist = model_merged_0.fit( X_train,
                            Y_train,
                            validation_data=[
                                X_val,
                                Y_val],
                            class_weight=class_weights,
                            callbacks=callbacks,
                            nb_epoch=500,
                            batch_size=32,
                            verbose=1)

    nb_epoch = len(hist.history['val_acc'])

    # nb_epoch = 33

    model_merged_1 = crnn_model(filter_density, pool_n_row, pool_n_col, nlen, dropout)
    model_merged_1.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    print(model_merged_1.count_params())

    hist = model_merged_1.fit(X_train_validation,
                                Y_train_validation,
                                nb_epoch=nb_epoch,
                                class_weight=class_weights,
                                batch_size=16)

    model_merged_1.save(file_path_model)

if __name__ == '__main__':

    import sys
    # feature_string = sys.argv[1]
    feature_string = 'timbre'
    file_path_model = '/scratch/rgongcrnnSingleSoundEvaluator_'+feature_string+'_gru1layer_10nodes_dense/out/crnn_model_cw_'+feature_string+'_nogru_dense.h5'

    X_train, X_val, X_train_validation, Y_train, Y_val, Y_train_validation, class_weights, max_length = \
    loadData(feature_string)

    train_model(filter_density=1,
                pool_n_row=3,
                pool_n_col=3,
                nlen = max_length,
                dropout=0.3,
                file_path_model=file_path_model,
                X_train=X_train,
                X_val=X_val,
                X_train_validation=X_train_validation,
                Y_train=Y_train,
                Y_val=Y_val,
                Y_train_validation=Y_train_validation,
                class_weights=class_weights)