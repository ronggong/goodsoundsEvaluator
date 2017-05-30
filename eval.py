from keras.models import load_model
import numpy as np
import pickle
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

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

if __name__ == '__main__':
    # filename_test_set = './dataset/testData_timbre.pkl'
    # filename_keras_model = './models/crnn_model_cw_timbre_gru1layer_32nodes_resnet_65k_bidirectional.h5'
    # testProcess(filename_test_set, filename_keras_model)

    # filename_test_set       = './dataset/testData_pitch.pkl'
    # filename_keras_model    = './models/crnn_model_cw_pitch_gru1layer_32nodes_resnet_65k_bidirectional.h5'
    # testProcess(filename_test_set, filename_keras_model)

    filename_test_set       = './dataset/testData_dynamics.pkl'
    filename_keras_model    = './models/crnn_model_cw_dynamics_gru1layer_32nodes_resnet_65k_bidirectional.h5'
    testProcess(filename_test_set, filename_keras_model)

    # filename_test_set       = './dataset/testData_richness.pkl'
    # filename_keras_model    = './models/crnn_model_cw_richness_gru1layer_32nodes_resnet_65k_bidirectional.h5'
    # testProcess(filename_test_set, filename_keras_model)

    # filename_test_set       = './dataset/testData_attack.pkl'
    # filename_keras_model    = './models/crnn_model_cw_attack_gru1layer_32nodes_resnet_65k_bidirectional.h5'
    # testProcess(filename_test_set, filename_keras_model)