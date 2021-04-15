# import os
# import numpy as np
# import sys
# import torch.utils.data
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import random
# import pandas as pd
#
#
# def read_ground_truth(building):
#     """
#     This implementation uses Soda as an example.
#     """
#     roomList = []
#     if building == "Soda":
#         f = open("./groundtruth/SODA-GROUND-TRUTH", "r+")
#         lines = f.readlines()
#         i = 0
#         while i < len(lines) - 1:
#             sensorName = lines[i].strip()
#             roomCorr = [sensorName]
#             i += 1
#             currLine = lines[i].strip()
#             # manually consider all cases.
#             if currLine.find("room-id") != -1:
#                 currLine = currLine.split(",")
#                 roomCorr.append(str(currLine[3]) + ", " + str(currLine[4]))
#                 # we need both room name and room id here
#                 roomList.append(roomCorr)
#             # elif currLine.find("chilled/condensor water loop-id") != -1:
#             #     currLine = currLine.split(",")
#             #     roomCorr.append(currLine[2])
#             #     roomList.append(roomCorr)
#             # elif currLine.find("supply fan-id") != -1:
#             #     currLine = currLine.split(",")
#             #     roomCorr.append(currLine[4])
#             #     roomList.append(roomCorr)
#             # elif currLine.find("ahu-id") != -1:
#             #     currLine = currLine.split(",")
#             #     roomCorr.append(currLine[2])
#             #     roomList.append(roomCorr)
#             # elif currLine.find("hot water loop-id") != -1:
#             #     currLine = currLine.split(",")
#             #     roomCorr.append(currLine[2])
#             #     roomList.append(roomCorr)
#             # elif currLine.find("chiller-id") != -1:
#             #     currLine = currLine.split(",")
#             #     roomCorr.append(currLine[2])
#             #     roomList.append(roomCorr)
#             # elif currLine.find("exhaust fan-id") != -1:
#             #     currLine = currLine.split(",")
#             #     try:
#             #         roomCorr.append(currLine[4])
#             #     except IndexError:
#             #         roomCorr.append(currLine[2])
#             #     roomList.append(roomCorr)
#             # elif currLine.find("condensor pump-id") != -1:
#             #     currLine = currLine.split(",")
#             #     roomCorr.append(currLine[2])
#             #     roomList.append(roomCorr)
#             else:
#                 pass
#
#             i += 1
#
#         f.close()
#     return roomList
#
#
# def clean_temperature(value):
#     for i in range(len(value)):
#         if value[i] > 40 or value[i] < 10:
#             if i == 0:
#                 value[i] = value[i + 1]
#             else:
#                 value[i] = value[i - 1]
#     return value
#
#
# def read_colocation_data(building, sensor_count, config):
#     x = []  # Store the value
#     y = []  # Store the room number
#     true_pos = []  # Store the filename
#     cnt = 0  # Index for list y
#     room_list = []  # Check if there is a sensor in the same room
#     groundTruth = read_ground_truth(building)
#     final_x, final_y, final_true_pos = [], [], []  # output
#
#     # Path depends on where the method is called (where the main is)
#     path = "rawdata/metadata/" + str(building)
#     # print(path)
#     folders = os.walk(path)
#
#     for path, dir_list, files in folders:
#         files.remove(".DS_Store")  # We don't want to read in .DS_Store file
#         for filename in files:  # Iterating through each time series file
#             if filename.endswith("csv"):
#                 _, value = read_csv(os.path.join(path, filename), config)
#                 # Have to clean the data in temperature sensors
#                 # But I do not know how to retrieve that information in SODA
#                 # Please change this to fit your code when editing
#                 if filename == 'temperature.csv':
#                     value = clean_temperature(value)
#
#             '''
#             Using name as the criteria for same room
#             Adding every room and name tuple into a list
#             If already in the list, find the corresponding room number
#             '''
#             if building == "Soda":
#
#                 filename = filename.strip(".csv")
#                 find = False  # whether we can find this sensor in groundTruth
#                 contains = False
#                 # whether this sensor is already contained in one of the rooms represented by elements in y
#                 currID = ""
#
#                 # checking whether this sensor is in the groundTruth
#                 for currSensor, currRoomID in groundTruth:
#                     if currSensor == filename:
#                         currID = currRoomID
#                         find = True
#
#                 # checking whether this sensor is in an existing room
#                 for sensor, tarSensorID, roomNumber in room_list:
#                     if currID == tarSensorID:
#                         contains = True
#                         y.append(roomNumber)
#                     else:
#                         pass
#
#                 if find:
#                     if contains:
#                         true_pos.append(filename)
#                     else:
#                         cnt += 1
#                         y.append(cnt)
#                         true_pos.append(filename)
#                         room_list.append([filename, currID, cnt])
#                     x.append(value)
#
#     # Only want four-sensor rooms
#
#     # Counting number of sensors in each room
#     countDict = {}
#     for index in y:
#         a = countDict.get(index)
#         if a is None:
#             countDict.update({index: 1})
#         else:
#             countDict.update({index: a + 1})
#
#     # Picking rooms with four sensors
#     wantedRoom = []
#     indexMap = {}
#     roomNum = 0
#     for key, value in countDict.items():
#         if value == sensor_count:
#             # Intuitively we should add the key into the list
#             # But the format requires the rooms to be indexes between 0 and length
#             # So we create a mapping between real key and the number we want
#             wantedRoom.append(key)
#             indexMap.update({key: roomNum})
#             roomNum += 1
#
#     # Adding desired rooms into output list
#     for i in range(len(y)):
#         if y[i] in wantedRoom:
#             final_x.append(x[i])
#             final_y.append(indexMap.get(y[i]))
#             final_true_pos.append(true_pos[i])
#
#     # sort lists to fit the format
#     zipped_list = zip(final_y, final_x, final_true_pos)
#     zipped_list = sorted(zipped_list)
#     final_y = [y for y, x, pos in zipped_list]
#     final_x = [x for y, x, pos in zipped_list]
#     final_true_pos = [pos for y, x, pos in zipped_list]
#
#     return final_x, final_y, final_true_pos
#
#
# def align_length(ts, val, maxl, sample_f=5):
#     if len(val) >= maxl:
#         return ts[0:maxl], val[0:maxl]
#     else:
#         for i in range(len(val), maxl):
#             val.append(0)
#             ts.append(ts[-1] + sample_f)
#         return ts, val
#
#
# def read_csv(path, config):
#     f = open(path)
#     timestamps, vals = [], []
#     for line in f.readlines():
#         if line == "":
#             pass
#         t, v = line.split(",")
#         timestamps.append(int(t))
#         vals.append(float(v))
#     return align_length(timestamps, vals, config.max_length)
#
#
# def sub_sample(ts, val, config):
#     sample_f = config.interval
#     MAXL = config.max_length
#
#     min_ts = ts[0]
#     max_ts = ts[-1]
#     new_ts, new_val = [], []
#     idx = 0
#     for t in range(min_ts, max_ts - sample_f, sample_f):
#         new_ts.append(t)
#         tmp, cnt = 0, 0
#         while ts[idx] < t + sample_f:
#             tmp += val[idx]
#             idx += 1
#             cnt += 1
#         if tmp != 0:
#             new_val.append(tmp / cnt)
#         else:
#             new_val.append(tmp)
#     return align_length(new_ts, new_val, MAXL, sample_f)
#
#
# def STFT(x, config):
#     fft_x = []
#     for i in range(len(x)):
#         fft_x.append(fft(x[i], config))
#     return fft_x
#
#
# def cross_validation_sample(total_cnt, test_cnt):
#     assert total_cnt % test_cnt == 0
#
#     folds = int(total_cnt / test_cnt)
#     idx = list(range(total_cnt))
#     random.shuffle(idx)
#     test_index = []
#     for i in range(folds):
#         fold_index = []
#         for j in range(test_cnt):
#             fold_index.append(idx[test_cnt * i + j])
#         test_index.append(fold_index)
#     return test_index
#
#
# def fft(v, config):
#     stride = config.stride
#     window_size = config.window_size
#     k_coefficient = config.k_coefficient
#     fft_data = []
#     fft_freq = []
#     power_spec = []
#     for i in range(int(len(v) / stride)):
#         if stride * i + window_size > len(v):
#             break
#         v0 = v[stride * i: stride * i + window_size]
#         v0 = np.array(v0)
#
#         fft_window = np.fft.fft(v0)[1:k_coefficient + 1]
#         fft_flatten = np.array([fft_window.real, fft_window.imag]).astype(np.float32).flatten('F')
#         fft_data.append(fft_flatten)
#
#     return np.transpose(np.array(fft_data))
#
#
# def split_colocation_train(x, y, test_index, split_method):
#     train_x, train_y, test_x, test_y = [], [], [], []
#     if split_method == 'room':
#         for i in range(len(y)):
#             if y[i] in test_index:
#                 test_x.append(x[i])
#                 test_y.append(y[i])
#             else:
#                 train_x.append(x[i])
#                 train_y.append(y[i])
#     else:
#         for i in range(len(y)):
#             if i not in test_index:
#                 train_x.append(x[i])
#                 train_y.append(y[i])
#             else:
#                 test_y.append(i)
#         test_x = x
#     return train_x, train_y, test_x, test_y
#
#
# def gen_colocation_triplet(train_x, train_y, prevent_same_type=False):
#     triplet = []
#     for i in range(len(train_x)):  # anchor
#         for j in range(len(train_x)):  # negative
#             if prevent_same_type and train_y[i] == train_y[j]:
#                 continue
#             for k in range(len(train_x)):  # positive
#                 if train_y[i] == train_y[j] or train_y[i] != train_y[k]:
#                     continue
#                 if i == k:
#                     continue
#                 # i != k and train_y[i]!= train_y[j] and train_y[i]==train_y[k]
#                 sample = [train_x[i], train_x[k], train_x[j]]
#                 triplet.append(sample)
#     return triplet
#
#
# def read_in_data(building, config):
#     # read data & STFT
#     x, y, true_pos = read_colocation_data(building, 4, config)
#     x = STFT(x, config)
#     return x, y, true_pos
#
#
# # From this point the codes remain unedited
# # Please check https://github.com/Shuheng-Li/Relational-Inference for more information
#
# def gen_coequipment_triplet(ahu_x, train_vav, ahu_y, train_y, mapping):
#     triplet = []
#
#     for i in range(len(train_vav)):  # anchor
#         k = ahu_y.index(mapping[train_y[i]])  # postive
#         for j in range(len(ahu_x)):  # negative
#             if j == k:
#                 continue
#             sample = []
#             sample.append(train_vav[i])
#             sample.append(ahu_x[k])
#             sample.append(ahu_x[j])
#             triplet.append(sample)
#     '''
#     for i in range(len(ahu_x)): #anchor
#         for j in range(len(train_vav)): #postive
#             for k in range(len(train_vav)): #negative
#                 if mapping[train_y[j]] != ahu_y[i] or mapping[train_y[k]] == ahu_y[i]:
#                     continue
#                 sample = []
#                 sample.append(ahu_x[i])
#                 sample.append(train_vav[k])
#                 sample.append(train_vav[j])
#                 triplet.append(sample)
#     '''
#     return triplet
#
#
# def clean_coequipment(ts, val, maxl=30000):
#     new_ts = [ts[0]]
#     new_val = [val[0]]
#     for i in range(1, len(ts)):
#         if ts[i] - ts[i - 1] == 1500:
#             new_val.append(val[i])
#         else:
#             k = int((ts[i] - ts[i - 1]) / 1500)
#             for _ in range(k):
#                 new_val.append(val[i])
#     return new_val[0:maxl]
#
#
# def read_ahu_csv(path, column=['PropertyTimestampInNumber', 'SupplyFanSpeedOutput']):
#     df = pd.read_csv(path)
#     ts = df[column[0]]
#     val = df[column[1]]
#     return clean_coequipment(ts, val)
#
#
# def read_vav_csv(path, column=['PropertyTimestampInNumber', 'AirFlowNormalized']):
#     df = pd.read_csv(path)
#     ts = df[column[0]]
#     val = df[column[1]]
#     return clean_coequipment(ts, val)
#
#
# def read_facility_ahu(facility_id, ahu_list):
#     ahu_data, label = [], []
#     path = "/localtmp/sl6yu/split/ahu_property_file_" + str(facility_id) + "/"
#     for name in ahu_list[facility_id]:
#         if os.path.exists(path + name + '.csv') == False:
#             continue
#         label.append(name)
#         ahu_data.append(read_ahu_csv(path + name + '.csv'))
#     return ahu_data, label
#
#
# def read_facility_vav(facility_id, mapping):
#     vav_data, label = [], []
#     path = "/localtmp/sl6yu/split/vav_box_property_file_" + str(facility_id) + "/"
#     for name in mapping[facility_id].keys():
#         if os.path.exists(path + name + '.csv') == False:
#             continue
#         label.append(name)
#         vav_data.append(read_vav_csv(path + name + '.csv'))
#     return vav_data, label
