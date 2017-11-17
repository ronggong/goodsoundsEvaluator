
# https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py

import numpy as np
import pickle
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.optimizers import Adam

from keras.layers import Dense, GRU, Bidirectional


def loadData(feature_string):
    # feature and label preparation
    # filename_trainData    = '../dataset/bag-of-feature-framelevel/trainData_'+feature_string+'.pkl'

    filename_trainData = '/scratch/rgongrnnSingleSoundEvaluator_' +\
                       feature_string + '_gru1layer_bidirectional_'+str(node_number)+\
                       'nodes/data/trainData_'+feature_string+'.pkl'

    with open(filename_trainData, 'rb') as f:
        X_train_validation, Y_train_validation  = pickle.load(f)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train_validation, Y_train_validation, test_size=0.2, stratify=Y_train_validation)

    class_weights = compute_class_weight('balanced',[0,1],Y_train_validation)
    # print(class_weights)
    class_weights = {0:class_weights[0], 1:class_weights[1]}

    # print(len(X_train), len(Y_train), len(X_val), len(Y_val))

    return X_train, X_val, Y_train, Y_val, class_weights


def rnn_model(node_number, include_top=True):
    model = Sequential()
    model.add(Bidirectional(GRU(node_number, return_sequences=False, name='gru2'), input_shape=(None, 94)))
    # model.add(Dropout(0.3))

    if include_top:
        model.add(Dense(1, activation='sigmoid', name='output'))

    return model

def modelEvaluation(model, feature, label):
    """
    evaluate binary classification model by output loss and accuracy
    :param model:
    :param feature:
    :param label:
    :return:
    """
    label_pred = []
    for x, y in zip(feature, label):
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x, batch_size=1, verbose=0)
        # print(val_pred)
        label_pred.append(pred[0][0])
    loss = log_loss(np.array(label), np.array(label_pred))

    label_pred = [0 if x < 0.5 else 1 for x in label_pred]
    acc = accuracy_score(np.array(label), np.array(label_pred))
    return loss, acc


def trainModelVariableLength(feature_train,
                             label_train,
                             feature_val,
                             label_val,
                             class_weight,
                             node_number,
                             nb_epoch,
                             patience,
                             file_path_model,
                             filename_out):

    model_0 = rnn_model(node_number)

    optimizer = Adam()

    model_0.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    model_0.summary()

    best_historical_val_loss = 1.0
    patience_earlyStopping = 0
    for jj in range(nb_epoch):
        ii = 0
        for x, y in zip(feature_train, label_train):
            # print x.shape, y, ii
            x = np.expand_dims(x, axis=0)
            model_0.fit(x, np.array([y]),
                        batch_size=1,
                        nb_epoch=1,
                        class_weight=class_weight,
                        verbose=0)
            ii += 1

        train_loss, train_acc = modelEvaluation(model_0, feature_train, label_train)
        val_loss, val_acc = modelEvaluation(model_0, feature_val, label_val)

        # write to the log
        if os.path.isfile(filename_out):
            file = open(filename_out, 'a')
        else:
            file = open(filename_out, 'w')

        file.write('epoch;'+str(jj)+';train_loss;'+str(train_loss)+';train_acc:'+str(train_acc)+
                   ';val_loss;'+str(val_loss)+';val_acc;'+str(val_acc) + '\n')
        file.close()

        # early stopping
        if val_loss < best_historical_val_loss:
            best_historical_val_loss = val_loss
            patience_earlyStopping = 0
            model_0.save(file_path_model)
        else:
            patience_earlyStopping += 1
            if patience_earlyStopping >= patience:
                break


if __name__ == '__main__':

    import sys
    feature_string = sys.argv[1]
    node_number = int(sys.argv[2])
    nb_epoch = int(sys.argv[3])
    # # feature_string = 'pitch'
    # # nb_epoch = 10
    file_path_model = '/scratch/rgongrnnSingleSoundEvaluator_'+\
                      feature_string+'_gru1layer_bidirectional_'+str(node_number)+\
                      'nodes/out/rnn_model_'+feature_string+'_gru1layer_bidirectional_'+\
                      str(node_number)+'nodes_variable_length.h5'


    filename_out = '/homedtic/rgong/noteEval/out/rnn_model_'\
                   +feature_string+'_gru1layer_bidirectional_'+\
                    str(node_number)+'nodes_variable_length.csv'

    if os.path.isfile(filename_out):
        file = open(filename_out, 'a')
    else:
        file = open(filename_out, 'w')
    # file.write("debug\n")
    file.close()

    X_train, X_val, Y_train, Y_val, class_weight = loadData(feature_string)
    trainModelVariableLength(X_train,
                             Y_train.tolist(),
                             X_val,
                             Y_val,
                             class_weight=class_weight,
                             node_number=node_number,
                             nb_epoch=nb_epoch,
                             patience=5,
                             file_path_model=file_path_model,
                             filename_out=filename_out)