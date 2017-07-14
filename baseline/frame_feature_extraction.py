import essentia.standard as ess
import pandas as pd
import numpy as np
from filePath import path_dataset,scaler_framelevel_path
from feature_extraction import extract_for_one
from xgb_classification import feature_scaling_train, feature_scaling_test
import os

def filenameClassExtraction(filename_csv):
    data = pd.DataFrame.from_csv(filename_csv)

    # the class label, either dan or laosheng
    y = data['class']

    filename_sound = data.index

    return filename_sound, y

def convert_pool_to_dataframe_framelevel(essentia_pool):
    '''
    convert essentia pool to dataframe
    :param essentia_pool:
    :param filename:
    :return: dataframe
    '''

    pool_dict = dict()
    for desc in essentia_pool.descriptorNames():
        # print(desc)
        # print(p[desc].shape)

        if essentia_pool[desc].ndim==1:
            pool_dict[desc] = essentia_pool[desc]
        elif essentia_pool[desc].ndim>1:
            dim_feature = essentia_pool[desc].shape[1]
            for ii in xrange(dim_feature):
                pool_dict[desc+str(ii)]=essentia_pool[desc][:,ii]
    return pd.DataFrame(pool_dict)


def framelevel_feature_extractor(feature_csv, str_train_test, jj):
    """
    extract feature in frame level,
    jj separates the dataset into chunks because of the memory issue
    :param feature_csv:
    :param str_train_test:
    :param jj:
    :return:
    """
    filename_sound, y = filenameClassExtraction(feature_csv)
    filename_sound = filename_sound

    for ii, fn in enumerate(filename_sound[1000*jj:1000*(jj+1)]):

        print(str_train_test, 1000*jj+ii, len(filename_sound))
        fn_noextension = os.path.splitext(fn)[0]
        fn_noextension = '-'.join(fn_noextension.split('/'))

        statsPool, p = extract_for_one(path_dataset, fn)
        dataframe_feature = convert_pool_to_dataframe_framelevel(p)
        dataframe_feature.to_csv(os.path.join(path_dataset,'feature_framelevel', fn_noextension+'.csv'))

def framelevel_label(feature_csv, str_train_test, dim_string):
    """
    save class label into a csv
    :param feature_csv:
    :param str_train_test:
    :param dim_string: timbre, richness, dynamics, attack, pitch
    :return:
    """
    filename_sound, y = filenameClassExtraction(feature_csv)
    filename_sound = filename_sound

    filename_sound_noextension = []
    for fn in filename_sound:
        fn_noextension = os.path.splitext(fn)[0]
        fn_noextension = '-'.join(fn_noextension.split('/'))
        filename_sound_noextension.append(fn_noextension)
    y.index = filename_sound_noextension
    y.to_csv(os.path.join(path_dataset,'feature_framelevel/y_'+str_train_test+'_'+dim_string+'.csv'))


def feature_scaling_framelevel_train(label_train_csv):
    label = pd.DataFrame.from_csv(label_train_csv)
    filename_train = label.index

    array_train = []
    for ii, fn in enumerate(filename_train):
        print('vstacking',ii,len(filename_train))
        data = pd.DataFrame.from_csv(os.path.join(path_dataset,'feature_framelevel',fn+'.csv'))
        X = data.values
        array_train.append(X)
    array_train = np.vstack(array_train)

    feature_scaling_train(array_train,'timbre',scaler_framelevel_path)
# print(y['class'])


# EXTRACTOR = ess.FreesoundExtractor()
#
# results = EXTRACTOR(filename_first_sound)

if __name__ == "__main__":
    # feature_timbre_train_csv = '../dataset/bag-of-feature/feature_timbre_train.csv'
    # framelevel_feature_extractor(feature_timbre_train_csv,'train',5)

    # feature_timbre_test_csv = '../dataset/bag-of-feature/feature_timbre_test.csv'
    # framelevel_feature_extractor(feature_timbre_test_csv,'test',1)

    label_train_csv = os.path.join(path_dataset,'feature_framelevel','y_train_timbre.csv')
    label_test_csv = os.path.join(path_dataset,'feature_framelevel','y_test_timbre.csv')

    # feature_scaling_framelevel_train(label_train_csv)

    # standarize feature for each sound
    label = pd.DataFrame.from_csv(label_train_csv, header=None)
    filename_train = label.index

    label = pd.DataFrame.from_csv(label_test_csv, header=None)
    filename_test = label.index
    for ii, fn in enumerate(filename_train+filename_test):

        print('vstacking',ii,len(filename_train+filename_test))

        data = pd.DataFrame.from_csv(os.path.join(path_dataset,'feature_framelevel',fn+'.csv'))
        X = data.values
        X = feature_scaling_test(X, 'timbre', scaler_framelevel_path)
        data[:] = X
        data.to_csv(os.path.join(path_dataset,'feature_framelevel_standarized',fn+'.csv'))