import numpy as np
import random
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# dimension order: [timbre, pitch, dynamics, richness, attack]

test_size = 0.2

feature_all, label_all = pickle.load(open('dataset/allData.pkl', 'rb'))

label_timbre_all    = [label_single_sound[0] for label_single_sound in label_all]
label_pitch_all     = [label_single_sound[1] for label_single_sound in label_all]
label_dynamics_all  = [label_single_sound[2] for label_single_sound in label_all]
label_richness_all  = [label_single_sound[3] for label_single_sound in label_all]
label_attack_all    = [label_single_sound[4] for label_single_sound in label_all]

print('total sample size:')
print(len(label_timbre_all))

print('bad sample size:')
print(len([ii for ii in label_timbre_all if ii == 0]))

def splitFromLabel(sss, label_all):
    """
    split features to train and test sets and return indices
    :param sss:
    :param label_all:
    :return:
    """
    for train_index, test_index in sss.split(feature_all, label_all):
        print(len(train_index))
        print(len(test_index))
        X_train = [feature_all[tpi] for tpi in train_index]
        y_train = [label_all[tpi] for tpi in train_index]

        X_test = [feature_all[tpi] for tpi in test_index]
        y_test = [label_all[tpi] for tpi in test_index]

    return X_train, X_test, y_train, y_test, train_index, test_index

def featureScaling(X_train, X_test):
    """
    Scaling features by X_train
    :param X_train:
    :param X_test:
    :return:
    """
    X_train_concat = np.concatenate(X_train, axis=0)
    print(X_train_concat.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train_concat)

    for ii in xrange(len(X_train)):
        X_train[ii] = scaler.transform(X_train[ii])

    for ii in xrange(len(X_test)):
        X_test[ii] = scaler.transform(X_test[ii])

    return X_train, X_test, scaler

def saveTestScaler(X_test, y_test, train_index, scaler, feature_string):
    pickle.dump((X_test, y_test), open('./dataset/testData_'+feature_string+'.pkl', 'wb'))
    pickle.dump(scaler, open('./dataset/scaler_'+feature_string+'_train.pkl', 'wb'))
    pickle.dump(train_index, open('./dataset/trainIndex_'+feature_string+'.pkl', 'wb'))

sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)

# sss.get_n_splits(feature_all, label_timbre_all)
# for train_timbre_index, test_timbre_index in sss.split(feature_all, label_timbre_all):
#     print(train_timbre_index)
#     print(test_timbre_index)

print('timbre')
sss.get_n_splits(feature_all, label_timbre_all)
X_timbre_train, X_timbre_test, y_timbre_train, y_timbre_test, train_timbre_index, test_timbre_index = splitFromLabel(sss, label_timbre_all)
_, X_timbre_test, scaler_timbre_train =  featureScaling(X_timbre_train, X_timbre_test)
saveTestScaler(X_timbre_test, y_timbre_test, train_timbre_index, scaler_timbre_train, 'timbre')

# # pitch
# print('pitch')
# sss.get_n_splits(feature_all, label_pitch_all)
# X_pitch_train, X_pitch_test, y_pitch_train, y_pitch_test, train_pitch_index, test_pitch_index = splitFromLabel(sss, label_pitch_all)
# _, X_pitch_test, scaler_pitch_train =  featureScaling(X_pitch_train, X_pitch_test)
# saveTestScaler(X_pitch_test, y_pitch_test, train_pitch_index, scaler_pitch_train, 'pitch')
#
# # dynamics
# print('dynamics')
#
# sss.get_n_splits(feature_all, label_dynamics_all)
# X_dynamics_train, X_dynamics_test, y_dynamics_train, y_dynamics_test, train_dynamics_index, test_dynamics_index = splitFromLabel(sss, label_dynamics_all)
# _, X_dynamics_test, scaler_dynamics_train =  featureScaling(X_dynamics_train, X_dynamics_test)
# saveTestScaler(X_dynamics_test, y_dynamics_test, train_dynamics_index, scaler_dynamics_train, 'dynamics')
#
# # richness
# print('richness')
#
# sss.get_n_splits(feature_all, label_richness_all)
# X_richness_train, X_richness_test, y_richness_train, y_richness_test, train_richness_index, test_richness_index = splitFromLabel(sss, label_richness_all)
# _, X_richness_test, scaler_richness_train =  featureScaling(X_richness_train, X_richness_test)
# saveTestScaler(X_richness_test, y_richness_test, train_richness_index, scaler_richness_train, 'richness')
#
# # attack
# print('attack')
#
# sss.get_n_splits(feature_all, label_attack_all)
# X_attack_train, X_attack_test, y_attack_train, y_attack_test, train_attack_index, test_attack_index = splitFromLabel(sss, label_attack_all)
# _, X_attack_test, scaler_attack_train =  featureScaling(X_attack_train, X_attack_test)
# saveTestScaler(X_attack_test, y_attack_test, train_attack_index, scaler_attack_train, 'attack')


# X_timbre_train, X_timbre_test, y_timbre_train, y_timbre_test = \
#     train_test_split(feature_all, label_timbre_all, test_size=test_size, stratify=label_timbre_all)
#
# print(len(X_timbre_train), len(X_timbre_test), len(y_timbre_train), len(y_timbre_test))
# print(X_timbre_train[0].shape)


# X_timbre_train, X_timbre_test, scaler = featureScaling(X_timbre_train, X_timbre_test)
#
# pickle.dump(scaler, open('./dataset/scaler_timbre_train.pkl', 'wb'))
# pickle.dump((X_timbre_train, y_timbre_train), open('./dataset/trainData_timbre.pkl', 'wb'))
# pickle.dump((X_timbre_test, y_timbre_test), open('./dataset/testData_timbre.pkl', 'wb'))