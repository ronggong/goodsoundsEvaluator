
# https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py

import numpy as np
import pickle
import random
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.preprocessing import sequence
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Dense, Reshape, Flatten, Merge, ELU, GRU
from keras.regularizers import l2


# feature and label preparation
# filename_all_set = 'dataset/allData.pkl'
filename_train_validation_set = '/scratch/rgongcrnnSingleSoundEvaluator_verbose/data/trainData.pkl'

# with open(filename_all_set, 'rb') as f:
#     X_all, Y_all  = pickle.load(f)

with open(filename_train_validation_set, 'rb') as f:
    X_train_validation, Y_train_validation  = pickle.load(f)


# print(X_train_validation[0].shape)

max_length = 1401
# for spectro in X_all:
#     if spectro.shape[0] > max_length:
#         max_length = spectro.shape[0]


num_val_set = int(0.2*len(X_train_validation))
idx_val_set = random.sample(xrange(len(X_train_validation)), num_val_set)

X_train = []
X_val = []

Y_train   = []
Y_val    = []


for ii in xrange(len(X_train_validation)):
    if ii in idx_val_set:
        X_val.append(sequence.pad_sequences(X_train_validation[ii].transpose(), maxlen=max_length, dtype='float32'))
        Y_val.append(Y_train_validation[ii])
    else:
        X_train.append(sequence.pad_sequences(X_train_validation[ii].transpose(), maxlen=max_length, dtype='float32'))
        Y_train.append(Y_train_validation[ii])

# print(X_train[0].shape)
# print(X_val[0].shape)
# print(len(X_train))

X_train = np.array(X_train, dtype='float32')
X_val = np.array(X_val, dtype='float32')

Y_train = np.array(Y_train, dtype='int64')
Y_val = np.array(Y_val, dtype='int64')

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)


def crnn_model(filter_density, pool_n_row, pool_n_col, nlen, dropout):

    reshape_dim = (80, nlen, 1)
    input_dim = (80, nlen)

    model = Sequential()

    # conv 0
    model.add(Reshape(reshape_dim, input_shape=input_dim))
    model.add(Convolution2D(int(64*filter_density), 3, 3, border_mode='valid',
                      input_shape=reshape_dim, dim_ordering='tf',
                      init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='tf'))
    model.add(Dropout(0.1))

    # conv 1
    model.add(Convolution2D(int(128*filter_density), 3, 3, border_mode='valid',
                            input_shape=reshape_dim, dim_ordering='tf',
                            init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', dim_ordering='tf'))
    model.add(Dropout(0.1))

    # conv 2
    model.add(Convolution2D(int(128 * filter_density), 3, 3, border_mode='valid',
                            input_shape=reshape_dim, dim_ordering='tf',
                            init='he_uniform', W_regularizer=l2(1e-5)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(3, 3), border_mode='valid', dim_ordering='tf'))
    model.add(Dropout(0.1))

    print(model.output_shape)

    # model.output_shape[1]*model.output_shape[2]

    model.add(Reshape((model.output_shape[1]*model.output_shape[2],128)))

    # GRU block 1, 2, output
    model.add(GRU(32, return_sequences=True, name='gru1'))
    model.add(GRU(32, return_sequences=False, name='gru2'))
    model.add(Dropout(0.3))

    model.add(Dense(5, activation='softmax', name='output'))

    optimizer = Adam()

    model.compile(loss='categorical_crossentropy',
                         optimizer=optimizer,
                         metrics=['accuracy'])

    return model

def train_model(filter_density, pool_n_row, pool_n_col, nlen, dropout, file_path_model):
    """
    train final model save to model path
    """

    model_merged_0 = crnn_model(filter_density, pool_n_row, pool_n_col, nlen, dropout)

    model_merged_0.summary()

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

    hist = model_merged_0.fit( X_train,
                            Y_train,
                            validation_data=[
                                X_val,
                                Y_val],
                            callbacks=callbacks,
                            nb_epoch=500,
                            batch_size=32,
                            verbose=1)

    nb_epoch = len(hist.history['val_acc'])

    model_merged_1 = crnn_model(filter_density, pool_n_row, pool_n_col, nlen, dropout)

    print(model_merged_1.count_params())

    hist = model_merged_1.fit(X_train_validation,
                                Y_train_validation,
                                nb_epoch=nb_epoch,
                                batch_size=32)

    model_merged_1.save(file_path_model)

if __name__ == '__main__':
    file_path_model = '/scratch/rgongcrnnSingleSoundEvaluator_verbose/out/crnn_model_0.h5'
    train_model(1, 3, 3, max_length, 0.3, file_path_model)