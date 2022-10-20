import pandas as pd
import random
import os
import numpy as np
from plastering.inferencers.relational_inference.util import logging, logging_result


# Data

def set_up_logging(config, args):
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + 'no_name' + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log = logging(log_path + 'log.txt')
    log_result = logging_result(log_path + 'record.txt')
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return log, log_result, log_path


def read_ground_truth(building):
    """
    This implementation uses Soda as an example.
    """
    roomList = []
    if building == "Soda":
        f = open("./groundtruth/SODA-GROUND-TRUTH", "r+")
        lines = f.readlines()
        i = 0
        while i < len(lines) - 1:
            sensorName = lines[i].strip()
            roomCorr = [sensorName]
            i += 1
            currLine = lines[i].strip()
            '''
            Manually consider all cases.
            If given more information, can rewrite in a more elegant way
            '''
            if currLine.find("room-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(str(currLine[3]) + ", " + str(currLine[4]))
                # we need both room name and room id here
                roomList.append(roomCorr)
            '''
            elif currLine.find("chilled/condensor water loop-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("supply fan-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[4])
                roomList.append(roomCorr)
            elif currLine.find("ahu-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("hot water loop-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("chiller-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("exhaust fan-id") != -1:
                currLine = currLine.split(",")
                try:
                    roomCorr.append(currLine[4])
                except IndexError:
                    roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            elif currLine.find("condensor pump-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(currLine[2])
                roomList.append(roomCorr)
            else:
                pass
            '''

            i += 1

        f.close()
    return roomList


def read_colocation_data(building, sensor_count, config):
    x = []  # Store the value
    y = []  # Store the room number
    true_pos = []  # Store the filename
    cnt = 0  # Index for list y
    room_list = []  # Check if there is a sensor in the same room
    groundTruth = read_ground_truth(building)
    final_x, final_y, final_true_pos = [], [], []  # output

    # Path depends on where the method is called (where the main is)
    path = "rawdata/metadata/" + str(building)
    # print(path)
    folders = os.walk(path)

    for path, dir_list, files in folders:
        files.remove(".DS_Store")  # We don't want to read in .DS_Store file
        for filename in files:  # Iterating through each time series file
            if filename.endswith("csv"):
                _, value = read_csv(os.path.join(path, filename), config)
                # Have to clean the data in temperature sensors
                # But I do not know how to retrieve that information in SODA
                # Please change this to fit your code when editing
                if filename == 'temperature.csv':
                    value = clean_temperature(value)

            '''
            Using name as the criteria for same room
            Adding every room and name tuple into a list
            If already in the list, find the corresponding room number
            '''
            if building == "Soda":

                filename = filename.strip(".csv")
                find = False  # whether we can find this sensor in groundTruth
                contains = False
                # whether this sensor is already contained in one of the rooms represented by elements in y
                currID = ""

                # checking whether this sensor is in the groundTruth
                for currSensor, currRoomID in groundTruth:
                    if currSensor == filename:
                        currID = currRoomID
                        find = True

                # checking whether this sensor is in an existing room
                for sensor, tarSensorID, roomNumber in room_list:
                    if currID == tarSensorID:
                        contains = True
                        y.append(roomNumber)
                    else:
                        pass

                if find:
                    if contains:
                        true_pos.append(filename)
                    else:
                        cnt += 1
                        y.append(cnt)
                        true_pos.append(filename)
                        room_list.append([filename, currID, cnt])
                    x.append(value)

    # Only want rooms with specific number of sensors

    # Counting number of sensors in each room
    countDict = {}
    for index in y:
        a = countDict.get(index)
        if a is None:
            countDict.update({index: 1})
        else:
            countDict.update({index: a + 1})

    # Picking rooms with sensor_count sensors
    wantedRoom = []
    indexMap = {}
    roomNum = 0
    for key, value in countDict.items():
        if value == sensor_count:
            # Intuitively we should add the key into the list
            # But the format requires the rooms to be indexes between 0 and length
            # So we create a mapping between real key and the number we want
            wantedRoom.append(key)
            indexMap.update({key: roomNum})
            roomNum += 1

    # Adding desired rooms into output list
    for i in range(len(y)):
        if y[i] in wantedRoom:
            final_x.append(x[i])
            final_y.append(indexMap.get(y[i]))
            final_true_pos.append(true_pos[i])

    # sort lists to fit the format
    zipped_list = zip(final_y, final_x, final_true_pos)
    zipped_list = sorted(zipped_list)
    final_y = [y for y, x, pos in zipped_list]
    final_x = [x for y, x, pos in zipped_list]
    final_true_pos = [pos for y, x, pos in zipped_list]

    return final_x, final_y, final_true_pos


def read_in_data(building, config):
    # read data & STFT
    x, y, true_pos = read_colocation_data(building, config.sensor_count, config)
    x = STFT(x, config)
    return x, y, true_pos


def cross_validation_sample(total_cnt, test_cnt):
    assert total_cnt % test_cnt == 0

    folds = int(total_cnt / test_cnt)
    idx = list(range(total_cnt))
    random.shuffle(idx)
    test_index = []
    for i in range(folds):
        fold_index = []
        for j in range(test_cnt):
            fold_index.append(idx[test_cnt * i + j])
        test_index.append(fold_index)
    return test_index


def clean_temperature(value):
    for i in range(len(value)):
        if value[i] > 40 or value[i] < 10:
            if i == 0:
                value[i] = value[i + 1]
            else:
                value[i] = value[i - 1]
    return value


def read_csv(path, config):
    f = open(path)
    timestamps, vals = [], []
    for line in f.readlines():
        if line == "":
            pass
        t, v = line.split(",")
        timestamps.append(int(t))
        vals.append(float(v))
    return align_length(timestamps, vals, config.max_length)


def align_length(ts, val, maxl, sample_f=5):
    if len(val) >= maxl:
        return ts[0:maxl], val[0:maxl]
    else:
        for i in range(len(val), maxl):
            val.append(0)
            ts.append(ts[-1] + sample_f)
        return ts, val


def STFT(x, config):
    fft_x = []
    for i in range(len(x)):
        fft_x.append(fft(x[i], config))
    return fft_x


def fft(v, config):
    stride = config.stride
    window_size = config.window_size
    k_coefficient = config.k_coefficient
    fft_data = []
    fft_freq = []
    power_spec = []
    for i in range(int(len(v) / stride)):
        if stride * i + window_size > len(v):
            break
        v0 = v[stride * i: stride * i + window_size]
        v0 = np.array(v0)

        fft_window = np.fft.fft(v0)[1:k_coefficient + 1]
        fft_flatten = np.array([fft_window.real, fft_window.imag]).astype(np.float32).flatten('F')
        fft_data.append(fft_flatten)

    return np.transpose(np.array(fft_data))


def split_colocation_train(x, y, true_pos, test_index, split_method):
    train_x, train_y, test_x, test_y = [], [], [], []
    train_true_pos, test_true_pos = [], []
    if split_method == 'room':
        for i in range(len(y)):
            if y[i] in test_index:
                test_x.append(x[i])
                test_y.append(y[i])
                test_true_pos.append(true_pos[i])
            else:
                train_x.append(x[i])
                train_y.append(y[i])
                train_true_pos.append(true_pos[i])
    else:
        for i in range(len(y)):
            if i not in test_index:
                train_x.append(x[i])
                train_y.append(y[i])
                train_true_pos.append(true_pos[i])
            else:
                test_y.append(i)
                test_true_pos.append(true_pos[i])
        test_x = x
    return train_x, train_y, train_true_pos, test_x, test_y, test_true_pos


def gen_colocation_triplet(train_x, train_y, prevent_same_type=False):
    triplet = []
    for i in range(len(train_x)):  # anchor
        for j in range(len(train_x)):  # negative
            if prevent_same_type and train_y[i] == train_y[j]:
                continue
            for k in range(len(train_x)):  # positive
                if train_y[i] == train_y[j] or train_y[i] != train_y[k]:
                    continue
                if i == k:
                    continue
                # i != k and train_y[i]!= train_y[j] and train_y[i]==train_y[k]
                sample = [train_x[i], train_x[k], train_x[j]]
                triplet.append(sample)
    return triplet


# coequipment


def clean_coequipment(ts, val):
    # timeArray = time.strptime(ts[0], "%m/%d/%Y %H:%M:%S")
    # new_ts = [int(time.mktime(timeArray))]
    new_val = []
    cnt = 0
    difs = []
    flag = False
    for i in range(len(val)):
        # timeArray = time.strptime(ts[i], "%m/%d/%Y %H:%M:%S")
        # new_ts.append(int(time.mktime(timeArray)))
        if np.isnan(val[i]):
            new_val.append(0)
        else:
            new_val.append(float(val[i]))
        if len(difs) > 5:
            flag = True
        elif new_val[-1] not in difs:
            difs.append(new_val[-1])
    return new_val, flag


def read_ahu_csv(path, column=['PropertyTimestamp', 'SupplyFanSpeedOutput']):
    # df = pd.read_csv(path)
    df = pd.read_csv(path, names=['PropertyTimestamp', 'SupplyFanSpeedOutput'])
    ts = df[column[0]]
    val = df[column[1]]
    return clean_coequipment(ts, val)


def read_vav_csv(path, column=['PropertyTimestamp', 'AirFlowNormalized']):
    # df = pd.read_csv(path)
    df = pd.read_csv(path, names=['PropertyTimestamp', 'SupplyFanSpeedOutput'])
    ts = df[column[0]]
    val = df[column[1]]
    return clean_coequipment(ts, val)


# read the ahu for a specific facility
def read_facility_ahu(facility_id, ahu_list, max_length):
    ahu_data, label = [], []
    # print(max_length)
    # column = ['PropertyTimestamp', ahu_s]

    # path = "/localtmp/split/ahu_property_file_" + str(facility_id) + "/"
    path = "./rawdata/metadata/" + str(facility_id) + "/"
    pops = []
    # print(column)
    for i, name in enumerate(ahu_list[facility_id]):
        if not os.path.exists(path + name + '.csv'):
            continue

        # ahu_d, flag = read_ahu_csv('/localtmp/split/ahu_property_file_' + str(facility_id) + '/' + name + '.csv')
        ahu_d, flag = read_ahu_csv(path + name + '.csv')
        if flag and len(ahu_d) >= max_length:
            ahu_data.append(ahu_d[0:max_length])
            label.append((facility_id, name))
            print(facility_id, name)
        else:
            pops.append(name)

    for name in pops:
        ahu_list[facility_id].pop(ahu_list[facility_id].index(name))
    return ahu_data, label


# read the vav for a specific facility
def read_facility_vav(facility_id, mapping, max_length, ahu_list):
    vav_data, label = [], []
    # column = ['PropertyTimestamp', vav_s]

    # path = "/localtmp/split/vav_box_property_file_" + str(facility_id) + "/"
    path = "./rawdata/metadata/" + str(facility_id) + "/"

    # print(column)
    for name in mapping[facility_id].keys():
        if not os.path.exists(path + name + '.csv'):
            continue
        if mapping[facility_id][name] not in ahu_list[facility_id]:
            continue

        # vav_d, flag = read_vav_csv('/localtmp/split/vav_box_property_file_' + str(facility_id) + '/' + name + '.csv')

        vav_d, flag = read_ahu_csv(path + name + '.csv')
        if flag and len(vav_d) >= max_length:
            vav_data.append(vav_d[0:max_length])
            label.append((facility_id, name))
            print(facility_id, name)
    return vav_data, label


# Original version
'''
def read_coequipment_ground_truth(path='./groundtruth/mapping_data.xlsx'):
    data = pd.read_excel(path, sheet_name='Hierarchy Data', usecols=[1, 6, 7, 9], engine='openpyxl')
    raw_list = data.values.tolist()
    mapping = dict()
    ahu_vas = dict()
    ahu_list = dict()
    for line in raw_list:
        # print(line)
        if line[3] != 'AHU':
            continue
        f_id = int(line[0])  # Facility_SID
        parent_name = line[1]  # PARENT_OBJECT_NAME
        child_name = line[2]  # CHILD_OBJECT_NAME
        if 'AHU-13  Area 112  MP581-4-4-2' == child_name:
            print("removed ")
            continue
        if f_id not in ahu_vas.keys():
            ahu_vas[f_id] = dict()
            ahu_list[f_id] = []
        ahu_list[f_id].append(child_name)
        ahu_vas[f_id][parent_name] = child_name

    for line in raw_list:
        if line[3] != 'VAV-BOX':
            continue
        f_id = int(line[0])
        parent_name = line[1]
        child_name = line[2]
        if f_id not in mapping.keys():
            mapping[f_id] = dict()
        if parent_name in ahu_vas[f_id].keys():
            # print(parent_name, ahu_vas[f_id])
            # print(ahu_vas[f_id][parent_name])
            # print(parent_name)
            # print(ahu_vas[f_id])
            mapping[f_id][child_name] = ahu_vas[f_id][parent_name]

            # print("------- ahu vas")
            # print(parent_name, ahu_vas[f_id])
            # print("------- mapping")
            # print(mapping[f_id])
            # print("-------")
            # print(child_name, mapping[f_id][child_name])
    return mapping, ahu_list
'''


# Adopted to fit building SODA
def read_coequipment_ground_truth():
    mapping = dict()
    ahu_list = dict()
    # Hard code this part
    mapping['Soda'] = dict()
    ahu_list['Soda'] = []
    mapping['Soda2'] = dict()
    ahu_list['Soda2'] = []

    sodaPath='./rawdata/metadata/soda_groud_truth'
    f = open(sodaPath, 'r+')
    lines = f.readlines()
    i = 0
    j = 0
    while i < len(lines) - 1:
        currLine = lines[i].strip("\n").split("\t\t")
        if currLine[1] == '12':
            ahu_list['Soda'].append(currLine[0])
            ahu_list['Soda2'].append(currLine[0])
        if currLine[1] == '4':
            for key in ahu_list['Soda']:
                # print(key)
                if currLine[0][0: 5] == key[0: 5]:
                    j += 1
                    if j % 2 == 0:
                        mapping['Soda'][currLine[0]] = key
                    else:
                        mapping['Soda2'][currLine[0]] = key
                    # mapping['Soda'][currLine[0]] = key
            # print(currLine[0])
        i += 1
    return mapping, ahu_list


def sub_sample(ts, val, config):
    sample_f = config.interval
    MAXL = config.max_length

    min_ts = ts[0]
    max_ts = ts[-1]
    new_ts, new_val = [], []
    idx = 0
    for t in range(min_ts, max_ts - sample_f, sample_f):
        new_ts.append(t)
        tmp, cnt = 0, 0
        while ts[idx] < t + sample_f:
            tmp += val[idx]
            idx += 1
            cnt += 1
        if tmp != 0:
            new_val.append(tmp / cnt)
        else:
            new_val.append(tmp)
    return align_length(new_ts, new_val, MAXL, sample_f)


def split_coequipment_train(vav_x, vav_y, test_index, train, test):
    train_vav, test_vav = [], []
    train_y, test_y = [], []
    shuffled_idx = np.arange(len(vav_x))
    np.random.shuffle(shuffled_idx)
    # print("test and train: ")
    # print(test, train)
    for i in shuffled_idx:
        if i in test_index:
            if vav_y[i][0] != test:
                continue
            test_vav.append(vav_x[i])
            test_y.append(vav_y[i])
        else:
            for item in train:
                if vav_y[i][0] == item:
                    train_vav.append(vav_x[i])
                    train_y.append(vav_y[i])
    return train_vav, train_y, test_vav, test_y


def gen_coequipment_triplet(ahu_x, ahu_y, vav_x, vav_y, mapping):
    # 1: a, p, n = (vav, ahu, ahu)
    # 2: a, p, n = (vav, ahu, vav)
    # 3: a, p, n = (vav, vav, ahu)
    # 4: a, p, n = (vav, vav, vav)
    # 5: a, p, n = (ahu, vav, vav)
    triplet = []
    # 1 (vav, ahu, ahu)
    # print(mapping[vav_y[0][0]])
    for i in range(len(vav_y)):  # anchor
        # print(mapping[vav_y[i][0]][vav_y[i][1]])
        k = ahu_y.index((vav_y[i][0], mapping[vav_y[i][0]][vav_y[i][1]]))  # positive
        for j in range(len(ahu_y)):  # negative
            if vav_y[i][0] != ahu_y[k][0] or vav_y[i][0] != ahu_y[j][0]:
                continue
            if j == k:
                continue
            # print(vav_y[i], ahu_y[k], ahu_y[j])
            sample = []
            sample.append(vav_x[i])
            sample.append(ahu_x[k])
            sample.append(ahu_x[j])
            triplet.append(sample)

    # 2 (vav, ahu, vav)
    '''
    for i in range(len(vav_y)): # anchor
        k = ahu_y.index((vav_y[i][0], mapping[vav_y[i][0]][vav_y[i][1]])) # positive
        for j in range(len(vav_y)): # negative
            if vav_y[i][0] != ahu_y[k][0] or vav_y[i][0] != vav_y[j][0]:
                continue
            if mapping[vav_y[i][0]][vav_y[i][1]] == mapping[vav_y[j][0]][vav_y[j][1]]:
                continue
            #print(vav_y[i], ahu_y[k], vav_y[j])
            sample = []
            sample.append(vav_x[i])
            sample.append(ahu_x[k])
            sample.append(vav_x[j])
            triplet.append(sample)
    

    # 3 (vav, vav, ahu)
    
    for i in range(len(vav_y)): # anchor
        for k in range(len(vav_y)): # positive
            for j in range(len(ahu_y)): # negative
                if vav_y[i][0] != vav_y[k][0] or vav_y[i][0] != ahu_y[j][0]:
                    continue
                if i == k or mapping[vav_y[i][0]][vav_y[i][1]] != mapping[vav_y[k][0]][vav_y[k][1]]:
                    continue
                if mapping[vav_y[i][0]][vav_y[i][1]] == ahu_y[j][1]:
                    continue
                #print(vav_y[i], vav_y[k], ahu_y[j])
                sample = []
                sample.append(vav_x[i])
                sample.append(vav_x[k])
                sample.append(ahu_x[j])
                triplet.append(sample)
    
    # 4 (vav, vav, vav)

    # 5 (ahu, vav, vav)
    
    for i in range(len(ahu_y)): # anchor
        for k in range(len(vav_y)): # positive
            for j in range(len(vav_y)): # negative
                if ahu_y[i][0] != vav_y[k][0] or ahu_y[i][0] != vav_y[j][0]:
                    continue
                if mapping[vav_y[k][0]][vav_y[k][1]] != ahu_y[i][1]:
                    continue
                if mapping[vav_y[j][0]][vav_y[j][1]] == ahu_y[i][1]:
                    continue
                #print(ahu_y[i], vav_y[k], vav_y[j])
                sample = []
                sample.append(ahu_x[i])
                sample.append(vav_x[k])
                sample.append(vav_x[j])
                triplet.append(sample)
    '''
    return triplet


def read_coequipment_data(config, args, test, train):
    # mapping, ahu_list = read_coequipment_ground_truth('./groundtruth/mapping_data.xlsx')
    mapping, ahu_list = read_coequipment_ground_truth()
    if config.all_facilities:
        facilities = [test]
        for f in train:
            facilities.append(f)
        ahu_x, ahu_y, vav_x, vav_y = [], [], [], []
        for f_id in facilities:
            f_ahu_x, f_ahu_y = read_facility_ahu(f_id, ahu_list, config.max_length)
            f_vav_x, f_vav_y = read_facility_vav(f_id, mapping, config.max_length, ahu_list)
            ahu_x += f_ahu_x
            ahu_y += f_ahu_y
            vav_x += f_vav_x
            vav_y += f_vav_y

        ahu_x = STFT(ahu_x, config)
        vav_x = STFT(vav_x, config)
    else:
        ahu_x, ahu_y = read_facility_ahu(args.facility, ahu_list, config.max_length)
        vav_x, vav_y = read_facility_vav(args.facility, mapping, config.max_length, ahu_list)
        ahu_x = STFT(ahu_x, config)
        vav_x = STFT(vav_x, config)

    logging("AHU %d total sensors, %d frequency coefficients, %d windows\n" % (
        len(ahu_x), ahu_x[0].shape[0], ahu_x[0].shape[1]))
    logging("VAV %d total sensors, %d frequency coefficients, %d windows\n" % (
        len(vav_x), vav_x[0].shape[0], vav_x[0].shape[1]))

    # split training & testing
    test_indices = cross_validation_sample(len(vav_y), int(len(vav_y) / 4))

    print("test indices:\n", test_indices)

    return ahu_x, ahu_y, vav_x, vav_y, test_indices, mapping
