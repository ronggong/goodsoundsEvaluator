import scipy.io as sio
import numpy as np
import pickle
import os
from xgb_classification import buildEstimators, prediction
from filePath import classifer_framelevel_path
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def grid_search(clf, X, y):
    # params = dict(anova_filter__k=['all'],
    #     # anova_filter__k=[50, 100, 'all'],
    #               xgb__max_depth=[3, 5, 10], xgb__n_estimators=[50, 100, 300, 500],
    #               xgb__learning_rate=[0.05, 0.1])
    n_iter_search = 20
    params = dict(anova_filter__k=[50, 100, 200, 'all'],
                  # anova_filter__k=[50, 100, 'all'],
                  xgb__max_depth=sp_randint(3, 10), xgb__n_estimators=sp_randint(10, 500),
                  xgb__learning_rate=[0.05, 0.1])
    # gs = GridSearchCV(clf, param_grid=params, n_jobs=4, cv=5, verbose=2)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    gs = RandomizedSearchCV(clf,
                            param_distributions=params,
                            n_jobs=4,
                            cv=cv,
                            n_iter=n_iter_search,
                            scoring='roc_auc',
                            verbose=2)
    gs.fit(X, y)

    print "Best estimator:"
    print gs.best_estimator_
    print "Best parameters:"
    print gs.best_params_
    print "Scorer:"
    print gs.scorer_
    print "Best score:"
    print gs.best_score_

    y_pred = gs.predict(X)
    y_test = y

    return gs.best_params_

def train_save(clf, X, y, dim_string, classifier_path):
    clf.fit(X, y)
    pickle.dump(clf, open(os.path.join(classifier_path,"xgb_classifier_"+dim_string+".plk"), "w"))

if __name__ == '__main__':
    # load i-vector feature
    filename_IVs = '/Users/gong/Documents/MATLAB/MTG/noteEvaluationIVectorFeatures/models/IVs_32mix.mat'
    mat = sio.loadmat(filename_IVs)

    print(mat['devIVsTrainPositive'].shape)
    print(mat['devIVsTrainNegative'].shape)
    print(mat['devIVsTestPositive'].shape)
    print(mat['devIVsTestNegative'].shape)

    devIVsTrainPositive = np.transpose(mat['devIVsTrainPositive'])
    devIVsTrainNegative = np.transpose(mat['devIVsTrainNegative'])

    nDims = devIVsTrainPositive.shape[1]

    nSamplesTrainPositive = devIVsTrainPositive.shape[0]
    nSamplesTrainNegative = devIVsTrainNegative.shape[0]

    devIVsTestPositive = np.transpose(mat['devIVsTestPositive'])
    devIVsTestNegative = np.transpose(mat['devIVsTestNegative'])

    nSamplesTestPositive = devIVsTestPositive.shape[0]
    nSamplesTestNegative = devIVsTestNegative.shape[0]

    dim_string = 'timbre'
    mode = 'test'

    # train a classifier using i-vector and test
    if mode == 'train':
        # feature train
        devIVsTrain = np.vstack([devIVsTrainPositive, devIVsTrainNegative])
        # label train
        yTrain = np.array([1] * nSamplesTrainPositive + [0] * nSamplesTrainNegative)

        print(devIVsTrain.shape)
        print(yTrain.shape)

        # initial xgb parameters
        init_params = {'xgb__learning_rate': 0.1, 'xgb__n_estimators': 500, 'xgb__max_depth': 5, 'anova_filter__k': 'all'}
        # xgb pipeline for grid search
        clf = buildEstimators(mode, dim_string, init_params, classifer_framelevel_path)
        best_params = grid_search(clf, devIVsTrain, yTrain)

        # xgb pipeline for best model
        clf = buildEstimators(mode, dim_string, best_params, classifer_framelevel_path)
        train_save(clf, devIVsTrain, yTrain, dim_string, classifer_framelevel_path)
    elif mode == 'test':
        # feature train
        devIVsTest = np.vstack([devIVsTestPositive, devIVsTestNegative])
        # label train
        yTest = np.array([1] * nSamplesTestPositive + [0] * nSamplesTestNegative)

        print(devIVsTest.shape)
        print(yTest.shape)

        clf = pickle.load(open(os.path.join(classifer_framelevel_path, "xgb_classifier_" + dim_string + ".plk"), "r"))
        prediction(clf, devIVsTest, yTest)
