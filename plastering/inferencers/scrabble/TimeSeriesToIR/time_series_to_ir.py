import csv
import numpy as np
import datetime
import time
from joblib import Parallel, delayed
import multiprocessing
import changefinder
import matplotlib.pyplot as plt
from feature_extractor import *
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
from ploting_classification_report import plot_classification_report
import pickle
import pdb

from randomizer import select_random_samples

class TimeSeriesToIR:

	def __init__(self, mlb=None, data_path="", model=RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=0)):
		self.mlb = mlb
		self.data_path = data_path
		self.model = model
		self.num_cores = multiprocessing.cpu_count()

	def train_model(self, features_path, cluster_filepath=None, training_percent=.4):

		with open(features_path, 'rb') as f:
			temp = pickle.load(f)	

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
			randomness = select_random_samples(cluster_filename=cluster_filepath, srcids=srcids, n=num_train, use_cluster_flag=1)

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
		report = classification_report(Y_test, Y_pred)
		plot_classification_report(report, self.mlb.classes_)

	def extractor(self, srcids, data_paths, schema_labels, num_points=5000, save_path="temp_var.pkl"):
		features = Parallel(n_jobs=self.num_cores)(delayed(features)(self.mlb, srcids[x], data_paths[x], schema_labels[x].lower().split(), num_points)
			for x in range(len(scrids)))

		with open(save_path, 'wb') as f:
			pickle.dump(features, f)
		
	def ts_to_ir (self, timeseries_filepath, num_points=5000):
		temp = Parallel(n_jobs=self.num_cores)(delayed(features)(self.mlb, None, timeseries_filepath[x], None, num_points)
			for x in range(len(timeseries_filepath)))
		Y_pred = self.model.predict(np.array(temp))
		temp_pred_proba = np.array(self.model.predict_proba(np.array(temp)))
		Y_proba = []
		for i in range(temp_pred_proba.shape[1]):
			temp = []
			for j in range(temp_pred_proba.shape[0]):
				temp.append(1 - temp_pred_proba[j][i][0])
			Y_proba.append(np.array(temp))
		Y_proba = np.array(Y_proba)

		return self.mlb.classes_, Y_pred, Y_proba



def features(mlb, srcid, filename, data_labels, num_points=None):
	reader = csv.reader(open(filename, "rb"), delimiter=",")
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

	if Y is None:
		return features
	else:
		return features, Y, srcid

