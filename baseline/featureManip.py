import pandas as pd
import json
from sklearn.model_selection import StratifiedShuffleSplit

def concatenatePd():
    filename_feature = '../dataset/bag-of-feature/bag-of-feature-0.csv'
    data_all = pd.DataFrame.from_csv(filename_feature)
    filename_data_all = '../dataset/bag-of-feature/bag-of-feature.csv'
    for ii in range(1,7):
        filename_feature = '../dataset/bag-of-feature/bag-of-feature-'+str(ii)+'.csv'
        data = pd.DataFrame.from_csv(filename_feature)
        data_all = pd.concat([data_all, data])

    data_all.to_csv(filename_data_all)

def splitFromLabel(sss, filename_all, label_all):
    """
    split features to train and test sets and return indices
    :param sss:
    :param label_all:
    :return:
    """
    for train_index, test_index in sss.split(filename_all, label_all):
        print(len(train_index))
        print(len(test_index))
        filename_train = [filename_all[tpi] for tpi in train_index]
        y_train = [label_all[tpi] for tpi in train_index]

        filename_test = [filename_all[tpi] for tpi in test_index]
        y_test = [label_all[tpi] for tpi in test_index]

    return filename_train, filename_test, y_train, y_test, train_index, test_index


def train_test_split():
    test_size = 0.2
    label_all = json.load(open('../test_oriol_4.json', 'rb'))

    filename_all = label_all.keys()

    annotation_all = [ii[0] for ii in label_all.values()]

    label_timbre_all = [label_single_sound[0] for label_single_sound in annotation_all]
    label_pitch_all = [label_single_sound[1] for label_single_sound in annotation_all]
    label_dynamics_all = [label_single_sound[2] for label_single_sound in annotation_all]
    label_richness_all = [label_single_sound[3] for label_single_sound in annotation_all]
    label_attack_all = [label_single_sound[4] for label_single_sound in annotation_all]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)

    sss.get_n_splits(filename_all, label_timbre_all)
    filename_timbre_train, filename_timbre_test, y_timbre_train, y_timbre_test, train_timbre_index, test_timbre_index = \
        splitFromLabel(sss, filename_all, label_timbre_all)

    sss.get_n_splits(filename_all, label_pitch_all)
    filename_pitch_train, filename_pitch_test, y_pitch_train, y_pitch_test, train_pitch_index, test_pitch_index = \
        splitFromLabel(sss, filename_all, label_pitch_all)

    sss.get_n_splits(filename_all, label_dynamics_all)
    filename_dynamics_train, filename_dynamics_test, y_dynamics_train, y_dynamics_test, train_dynamics_index, test_dynamics_index = \
        splitFromLabel(sss, filename_all, label_dynamics_all)

    sss.get_n_splits(filename_all, label_richness_all)
    filename_richness_train, filename_richness_test, y_richness_train, y_richness_test, train_richness_index, test_richness_index = \
        splitFromLabel(sss, filename_all, label_richness_all)

    sss.get_n_splits(filename_all, label_attack_all)
    filename_attack_train, filename_attack_test, y_attack_train, y_attack_test, train_attack_index, test_attack_index = \
        splitFromLabel(sss, filename_all, label_attack_all)

    timbre_split    = [filename_timbre_train, filename_timbre_test, y_timbre_train, y_timbre_test]
    pitch_split     = [filename_pitch_train, filename_pitch_test, y_pitch_train, y_pitch_test]
    dynamics_split  = [filename_dynamics_train, filename_dynamics_test, y_dynamics_train, y_dynamics_test]
    richness_split  = [filename_richness_train, filename_richness_test, y_richness_train, y_richness_test]
    attack_split    = [filename_attack_train, filename_attack_test, y_attack_train, y_attack_test]


    filename_data_all   = '../dataset/bag-of-feature/bag-of-feature.csv'
    data_all            = pd.DataFrame.from_csv(filename_data_all)

    feature_timbre_train    = data_all.loc[timbre_split[0],:]
    feature_timbre_test     = data_all.loc[timbre_split[1],:]
    feature_timbre_train['class'] = pd.Series(timbre_split[2], index=feature_timbre_train.index)
    feature_timbre_test['class'] = pd.Series(timbre_split[3], index=feature_timbre_test.index)
    feature_timbre_train.to_csv('../dataset/bag-of-feature/feature_timbre_train.csv')
    feature_timbre_test.to_csv('../dataset/bag-of-feature/feature_timbre_test.csv')

    feature_pitch_train    = data_all.loc[pitch_split[0],:]
    feature_pitch_test     = data_all.loc[pitch_split[1],:]
    feature_pitch_train['class'] = pd.Series(pitch_split[2], index=feature_pitch_train.index)
    feature_pitch_test['class'] = pd.Series(pitch_split[3], index=feature_pitch_test.index)
    feature_pitch_train.to_csv('../dataset/bag-of-feature/feature_pitch_train.csv')
    feature_pitch_test.to_csv('../dataset/bag-of-feature/feature_pitch_test.csv')

    feature_dynamics_train    = data_all.loc[dynamics_split[0],:]
    feature_dynamics_test     = data_all.loc[dynamics_split[1],:]
    feature_dynamics_train['class'] = pd.Series(dynamics_split[2], index=feature_dynamics_train.index)
    feature_dynamics_test['class'] = pd.Series(dynamics_split[3], index=feature_dynamics_test.index)
    feature_dynamics_train.to_csv('../dataset/bag-of-feature/feature_dynamics_train.csv')
    feature_dynamics_test.to_csv('../dataset/bag-of-feature/feature_dynamics_test.csv')

    feature_richness_train    = data_all.loc[richness_split[0],:]
    feature_richness_test     = data_all.loc[richness_split[1],:]
    feature_richness_train['class'] = pd.Series(richness_split[2], index=feature_richness_train.index)
    feature_richness_test['class'] = pd.Series(richness_split[3], index=feature_richness_test.index)
    feature_richness_train.to_csv('../dataset/bag-of-feature/feature_richness_train.csv')
    feature_richness_test.to_csv('../dataset/bag-of-feature/feature_richness_test.csv')

    feature_attack_train    = data_all.loc[attack_split[0],:]
    feature_attack_test     = data_all.loc[attack_split[1],:]
    feature_attack_train['class'] = pd.Series(attack_split[2], index=feature_attack_train.index)
    feature_attack_test['class'] = pd.Series(attack_split[3], index=feature_attack_test.index)
    feature_attack_train.to_csv('../dataset/bag-of-feature/feature_attack_train.csv')
    feature_attack_test.to_csv('../dataset/bag-of-feature/feature_attack_test.csv')

train_test_split()
