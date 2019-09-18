from time_series_to_ir import *
from random import shuffle
import pickle

with open("temp/ebu3b_features.pkl", 'rb') as f:
	ebu3b_features = pickle.load(f)

with open("Binarizer/mlb.pkl", 'rb') as f:
	mlb = pickle.load(f)


with open("Model/model_random_forest.pkl", 'rb') as f:
	trained_model = pickle.load(f)
trained_model.warm_start = True

ebu3b_srcids =[]
ebu3b_data = []
ebu3b_binarized_label = []
for i in ebu3b_features:
	ebu3b_srcids.append(i[2])
	ebu3b_data.append(i[0])
	ebu3b_binarized_label.append(i[1])

index = range(len(ebu3b_srcids))
shuffle(index)

time_series_to_ir = TimeSeriesToIR(mlb=mlb, model=trained_model)
mlb_keys, Y_pred, Y_proba = time_series_to_ir.ts_to_ir(train_features=ebu3b_features, train_srcids=[ebu3b_srcids[i] for i in index[:100]]
	, test_features=ebu3b_features, test_srcids=[ebu3b_srcids[i] for i in index[100:103]])
print(Y_proba)
print(Y_pred)
