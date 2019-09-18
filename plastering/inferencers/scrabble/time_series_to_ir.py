import csv
import numpy as np
import datetime
import time
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import pdb

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn import preprocessing
from random import shuffle
from sklearn.multiclass import *
from sklearn.externals import joblib
from sklearn.svm import *
from sklearn import tree
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_fscore_support
import pickle
import binascii

from .feature_extractor import *
#from .randomizer import select_random_samples
from .ploting_classification_report import plot_classification_report

class TimeSeriesToIR:

    def __init__(self, mlb=None, model=RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=0)):
        self.mlb = mlb
        self.model = model
        self.num_cores = multiprocessing.cpu_count()
        self.classes_indx = []

    def get_binarizer(self):
        return self.mlb

    def train_model(self, features, include_feature=None, cluster_filepath=None, training_percent=.4):
        temp = []
        if(include_feature is None):
            temp = features
        else:
            for i in features:
                data = []
                for j in include_feature:
                    data.append(i[0][j])
                data = np.array(data)
                temp.append([data, i[1], i[2]])


        index = range(len(temp))
        num_train = int(training_percent * float(len(index)))

        srcids = []

        for i in index:
            srcids.append(temp[i][2])

        if(cluster_filepath is None):
            X = list()
            Y = list()
            index = np.subtract(np.array(index), 1)
            for a in temp:
                X.append(normalize(np.array(a[0]).reshape((1,-1))).reshape(-1))
                Y.append(a[1][0])
            Y = np.array(Y)
            X = np.array(X)
            shuffle(index)
            X_train, X_test = X[index[:num_train]], X[index[num_train:]]
            Y_train, Y_test = Y[index[:num_train]], Y[index[num_train:]]
        else:
            X_train, X_test = list(), list()
            Y_train, Y_test = list(), list()
            randomness = select_random_samples(
                cluster_filename=cluster_filepath,
                srcids=[x for x in srcids],
                n=num_train,
                use_cluster_flag=1)
            for i in range(len(index)):
                not_in_set = True
                for j in range(len(randomness)):
                    if(temp[i][2] == randomness[j]):
                        X_train.append(normalize(np.array(temp[i][0]).reshape((1,-1))).reshape(-1))
                        Y_train.append(temp[i][1][0])
                        not_in_set = False
                if(not_in_set):
                    X_test.append(normalize(np.array(temp[i][0]).reshape((1,-1))).reshape(-1))
                    Y_test.append(temp[i][1][0])
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            Y_train = np.array(Y_train)
            Y_test = np.array(Y_test)
        print(X_train.shape)
        print(Y_train.shape)
        self.model.fit(X_train, Y_train)
        Y_pred = self.model.predict(X_test)
        print(include_feature)
        print("Weighted ", precision_recall_fscore_support(Y_test, Y_pred, average='weighted'))
        print("Macro ", precision_recall_fscore_support(Y_test, Y_pred, average='macro'))
        print("Micro ", precision_recall_fscore_support(Y_test, Y_pred, average='micro'))
        print("Samples ", precision_recall_fscore_support(Y_test, Y_pred, average='samples'))

    def feature_extractor(self, srcids, schema_labels, data_path="data/", num_points=5000):
        return Parallel(n_jobs=self.num_cores)(delayed(features)(self.mlb, srcids[x], schema_labels[x].lower().split(), data_path, num_points)
            for x in range(len(srcids)))

    def fit(self, train_features, train_srcids,
                  val_srcids, train_tags_dict=None):
        self.ts_to_ir(train_features, train_srcids,
                       train_features, val_srcids, True)

    def predict(self, test_features, test_srcids):
        return self.ts_to_ir(train_features=None,
                             train_srcids=None,
                             test_features=test_features,
                             test_srcids=test_srcids, 
                             val=False)

    def ts_to_ir (self, train_features=None, train_srcids=None, test_features=None, test_srcids=None, val=False):
        if train_srcids:
            X_train = []
            Y_train = []
            for i in train_srcids:
                for j in train_features:
                    if(j[2] == i):
                        X_train.append(normalize(np.array(j[0]).reshape((1,-1))).reshape(-1))
                        Y_train.append(j[1][0])
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            print(X_train.shape)
            print(Y_train.shape)
            self.model.fit(X_train, Y_train)

        if test_srcids:
            test_data = []

            Y_test = []
            for i in test_srcids:
                for j in test_features:
                    if(j[2] == i):
                        test_data.append(normalize(np.array(j[0]).reshape((1,-1))).reshape(-1))
                        Y_test.append(np.asarray([j[1][0]]))
            Y_pred = self.model.predict(np.array(test_data))
            if val:
                Y_test = np.vstack(Y_test)
                report = classification_report(Y_test, Y_pred)
                lines_val = report.split('\n')

                self.classes_indx = []
                for i in range(len(lines_val[2 : (len(lines_val) - 2)])):
                    t = lines_val[i+2].strip().split()
                    if len(t) < 2: continue
                    # print(float(t[-2]))
                    if float(t[1]) < 0.7: continue
                    self.classes_indx.append(int(t[0]))
            else:
                print(self.mlb.classes_[self.classes_indx])
                # print(range(len(Y_pred)))
                for temp in Y_pred:
                    # print(len(temp))
                    for i in range(len(temp)):
                        if i not in self.classes_indx:
                            temp[i] = 0
                # print(v)
            # temp_pred_proba = self.model.predict_proba(np.array(test_data))
            # Y_proba = []
            # for i in range(temp_pred_proba.shape[1]):
            #   temp = []
            #   for j in range(temp_pred_proba.shape[0]):
            #       temp.append(1 - temp_pred_proba[j][i][0])
            #   Y_proba.append(np.array(temp))
            # Y_proba = np.array(Y_proba)

            #return self.mlb.classes_, Y_pred, np.array(Y_test)#, Y_proba
            return Y_pred

    def ploting(self, Y_pred_val, Y_val, Y_pred_test, Y_test, fig_path="temp.png"):
        report_val = classification_report(Y_val, Y_pred_val)
        report_test = classification_report(Y_test, Y_pred_test)
        plot_classification_report(report_val, report_test, self.mlb.classes_, fig_path=fig_path)


def features(mlb, srcid, data_labels, file_path="data/", num_points=None):
    reader = csv.reader(open(file_path + srcid +".csv", "r"), delimiter=",")
    file_data = list(reader)
    temp = list()
    for y in file_data[1:]:
        temp.append(y[1])
    data = np.array(temp, dtype="float")
    if(data_labels is None):
        Y = None
    else:
        Y = mlb.transform([set(data_labels)])

    if num_points is None:
        features = get_features(data)
    elif num_points >= len(data):
        features = get_features(data)
    else:
        features = get_features(data[-num_points:])

    return features, Y, srcid

