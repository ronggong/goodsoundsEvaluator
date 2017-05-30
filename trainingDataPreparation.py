import essentia.standard as ess
import numpy as np
from parameters import *


def getMFCCBands2D(audio, framesize):

    '''
    mel bands feature [p[0],p[1]]
    output feature for each time stamp is a 2D matrix
    it needs the array format float32
    :param audio:
    :param p:
    :return:
    '''

    winAnalysis = 'hann'

    MFCC80 = ess.MFCC(sampleRate=fs,
                      highFrequencyBound=highFrequencyBound,
                      inputSize=framesize + 1,
                      numberBands=80)

    N = 2 * framesize  # padding 1 time framesize
    SPECTRUM = ess.Spectrum(size=N)
    WINDOW = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC80(mXFrame)
        mfcc.append(bands)

    # the mel bands features
    feature = np.array(mfcc,dtype='float32')

    return feature

# def trainingDataPreparation(audio, framesize, )

if __name__ == '__main__':


    import json
    import pickle
    from filePath import *
    from os.path import join


    # label_oriol = json.load(open('test_oriol_4.json', 'rb'))
    #
    # feature_all = []
    # label_all   = []
    #
    # for ii_fn, fn in enumerate(label_oriol):
    #
    #     print('doing calculation for '+fn, ii_fn, len(label_oriol))
    #
    #     full_path_fn    = join(path_dataset, fn)
    #     label           = label_oriol[fn][0]
    #
    #     audio               = ess.MonoLoader(downmix = 'left', filename = full_path_fn, sampleRate = fs)()
    #     feature             = getMFCCBands2D(audio, framesize)
    #     feature             = np.log(100000 * feature + 1)
    #
    #     feature_all.append(feature)
    #     label_all.append(label)
    #
    # label_all = np.array(label_all,dtype='int64')
    #
    # print(len(feature_all), label_all.shape)
    #
    # pickle.dump((feature_all, label_all), open('allData.pkl', 'wb'))


    import random
    from sklearn import preprocessing

    feature_all, label_all = pickle.load(open('allData.pkl', 'rb'))

    num_test_set = int(0.2*len(feature_all))

    idx_test_set = random.sample(xrange(len(feature_all)), num_test_set)

    feature_train = []
    feature_test  = []

    label_train   = []
    label_test    = []

    for ii in xrange(len(feature_all)):
        if ii in idx_test_set:
            feature_test.append(feature_all[ii])
            label_test.append(label_all[ii,:])
        else:
            feature_train.append(feature_all[ii])
            label_train.append(label_all[ii,:])

    feature_train_concat = np.concatenate(feature_train, axis=0)
    print(feature_train_concat.shape)

    scaler = preprocessing.StandardScaler()
    scaler.fit(feature_train_concat)

    for ii in xrange(len(feature_train)):
        feature_train[ii] = scaler.transform(feature_train[ii])

    for ii in xrange(len(feature_test)):
        feature_test[ii] = scaler.transform(feature_test[ii])

    pickle.dump(scaler, open('scaler_train.pkl', 'wb'))
    pickle.dump((feature_train, label_train), open('trainData.pkl', 'wb'))
    pickle.dump((feature_test, label_test), open('testData.pkl', 'wb'))