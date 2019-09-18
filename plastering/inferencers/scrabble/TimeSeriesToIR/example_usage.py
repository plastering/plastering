from time_series_to_ir import *
import pickle


with open("mlb.pkl", 'rb') as f:
	mlb = pickle.load(f)


with open("model.pkl", 'rb') as f:
	trained_model = pickle.load(f)

time_series_to_ir = TimeSeriesToIR(mlb=mlb, model=trained_model)
print(time_series_to_ir.ts_to_ir(["../data/505_0_3000002.csv", "../data/505_0_3000112.csv"]))
