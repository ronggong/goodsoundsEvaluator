
import pandas as pd
import numpy
import pickle
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
from sklearn.cross_validation import StratifiedKFold, train_test_split
import xgboost as xgb

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from filePath import *
# from utils import visualization


# IRMAS - 30
THRESHOLD = 20

labels = None


def feature_preprocessing(datafile):

    # import some data to play with
    data = pd.DataFrame.from_csv(datafile)

    # the class label, either dan or laosheng
    y = data['class']

    # feature matrix, axis 0: observation, axis 1: feature dimension
    X = data.drop(['class'], axis=1).values

    return X, y

def feature_scaling_train(X, dim_string):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    pickle.dump(scaler,open(path.join(scaler_path,'feature_scaler_'+dim_string+'.pkl'),'wb'))
    # X = scaler.transform(X)

    return

def feature_scaling_test(X, dim_string):
    scaler = pickle.load(open(path.join(scaler_path,'feature_scaler_'+dim_string+'.pkl'),'r'))
    X = scaler.transform(X)

    return X

def buildEstimators(mode, dim_string, best_params):
    if mode == 'train' or mode == 'cv':
        # best parameters got by gridsearchCV, best score: 1
        estimators = [('anova_filter', SelectKBest(f_classif, k=best_params['anova_filter__k'])),
                      ('xgb', xgb.XGBClassifier(learning_rate=best_params['xgb__learning_rate'],
                                                n_estimators=best_params['xgb__n_estimators'],
                                                max_depth=best_params['xgb__max_depth']))]
        clf = Pipeline(estimators)
    elif mode == 'test':
        clf = pickle.load(open(join(classifier_path,"xgb_classifier_"+dim_string+".plk"), "r"))
    return clf

def imputerLabelEncoder_train(X,y):
    imputer = preprocessing.Imputer()
    X = imputer.fit_transform(X)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return X,y,imputer,le

def imputerLabelEncoder_test(X,y,dim_string):
    imputer = pickle.load(open(join(classifier_path,"inputer_"+dim_string+".plk"),'r'))
    X = imputer.fit_transform(X)

    le = pickle.load(open(join(classifier_path,"le_"+dim_string+".plk"), "r"))
    y = le.fit_transform(y)
    return X,y

def imputer_run(X,dim_string):
    imputer = pickle.load(open(join(classifier_path,"inputer_"+dim_string+".plk"),'r'))
    X = imputer.fit_transform(X)
    return X

def save_results(y_test, y_pred, labels, fold_number=0):
    pickle.dump(y_test, open("y_test_fold{number}.plk".format(number=fold_number), "w"))
    pickle.dump(y_pred, open("y_pred_fold{number}.plk".format(number=fold_number), "w"))
    print classification_report(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)
    print "Micro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='micro')
    print "Macro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='macro')
    try:
        visualization.plot_confusion_matrix(confusion_matrix(y_test, y_pred),
                                            title="Test CM fold{number}".format(number=fold_number),
                                            labels=labels)
    except:
        pass


def train_test(clf, X, y, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    save_results(y_test, y_pred, labels)

def train_evaluate_stratified(clf, X, y, labels):
    skf = StratifiedKFold(y, n_folds=10)
    for fold_number, (train_index, test_index) in enumerate(skf):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        save_results(y_test, y_pred, labels, fold_number)


def grid_search(clf, X, y):
    # params = dict(anova_filter__k=['all'],
    #     # anova_filter__k=[50, 100, 'all'],
    #               xgb__max_depth=[3, 5, 10], xgb__n_estimators=[50, 100, 300, 500],
    #               xgb__learning_rate=[0.05, 0.1])
    n_iter_search = 20
    params = dict(anova_filter__k=[50, 100, 'all'],
                  # anova_filter__k=[50, 100, 'all'],
                  xgb__max_depth=sp_randint(3, 10), xgb__n_estimators=sp_randint(10, 500),
                  xgb__learning_rate=[0.05])
    # gs = GridSearchCV(clf, param_grid=params, n_jobs=4, cv=5, verbose=2)
    gs = RandomizedSearchCV(clf, param_distributions=params,n_jobs=4,cv=5,n_iter=n_iter_search,verbose=2)
    gs.fit(X, y)

    print "Best estimator:"
    print gs.best_estimator_
    print "Best parameters:"
    print gs.best_params_
    print "Best score:"
    print gs.best_score_

    y_pred = gs.predict(X)
    y_test = y

    return gs.best_params_

def train_save(clf, X, y, le, inputer,dim_string):
    clf.fit(X, y)
    pickle.dump(clf, open(join(classifier_path,"xgb_classifier_"+dim_string+".plk"), "w"))
    pickle.dump(le, open(join(classifier_path,"le_"+dim_string+".plk"), "w"))
    pickle.dump(inputer, open(join(classifier_path,"inputer_"+dim_string+".plk"), "w"))

def prediction(clf, X, y):
    y_pred = clf.predict(X)
    y_test = y
    print classification_report(y_test, y_pred)
    # print confusion_matrix(y_test, y_pred)
    print "Accuracy:"
    print accuracy_score(y_test, y_pred)
    print "Micro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='micro')
    print "Macro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='macro')

if __name__ == "__main__":
    datafile = sys.argv[1]
    mode=sys.argv[2]
    dim_string=sys.argv[3]

    X, y = feature_preprocessing(datafile)

    X[numpy.isinf(X)] = numpy.iinfo('i').max

    if mode == 'train' or mode == 'cv':
        feature_scaling_train(X,dim_string)

    X = feature_scaling_test(X,dim_string)

    if mode == 'train' or mode == 'cv':
        X,y,imputer,le = imputerLabelEncoder_train(X,y)
    elif mode == 'test':
        X,y = imputerLabelEncoder_test(X,y,dim_string)

    # print X,y
    best_params = {'xgb__learning_rate': 0.1, 'xgb__n_estimators': 500, 'xgb__max_depth': 5, 'anova_filter__k': 'all'}
    clf = buildEstimators(mode, dim_string, best_params)
    if mode == 'cv' or mode == 'train':

        best_params = grid_search(clf,X,y)
        clf = buildEstimators(mode, dim_string, best_params)
        train_save(clf, X, y, le, imputer,dim_string)
    elif mode == 'test':
        prediction(clf, X, y)
