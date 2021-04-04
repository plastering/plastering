import argparse
import os
import numpy as np
import yaml
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# TODO: Add to requirement


# util
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))


def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s)

    return write_log


def logging_result(file):
    def write_log(s):
        with open(file, 'a') as f:
            f.write(s)

    return write_log


# # TODO: Change this
# def cal_sensor_acc(best_solution, test_y):
#     total, cnt = 0, 0
#     for i in range(len(best_solution)):
#         for j in range(len(best_solution[i]) - 1):
#             for k in range(j + 1, len(best_solution[i])):
#                 # print(best_solution[i][j], best_solution[i][k])
#                 if best_solution[i][j] in test_y or best_solution[i][k] in test_y:
#                     total += 1
#                     print(best_solution[i][j], best_solution[i][k])
#                 else:
#                     continue
#                 if int(best_solution[i][j] / 4) == int(best_solution[i][k] / 4):
#                     cnt += 1
#     # print(total, cnt)
#     acc = cnt / total
#     return acc


# TODO: Change this
def cal_room_acc(best_solution):
    pp, pn, np, nn = 0, 0, 0, 0  # (ground_truth, prediction)
    for i in range(len(best_solution)):
        for j in range(len(best_solution[i]) - 1):
            for k in range(j + 1, len(best_solution[i])):
                if int(best_solution[i][j] / 4) == int(best_solution[i][k] / 4):
                    pp += 1
                else:
                    pn += 1
                    np += 1
    nn = (len(best_solution) * len(best_solution[0])) * (
            len(best_solution) * len(best_solution[0]) - 1) / 2 - pp - pn - np
    recall = pp / (pp + pn)
    acc_room = 0
    for i in range(len(best_solution)):
        r_id = int(best_solution[i][0] / 4)
        for j in range(1, 5):
            if j == 4:
                acc_room += 1
                break
            if int(best_solution[i][j] / 4) != r_id:
                break
    room_wise_acc = acc_room / len(best_solution)
    confusion = [[pp, np], [pn, nn]]
    return recall, room_wise_acc


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


# TODO: Handle different buildings
def read_ground_truth(building):
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
            # manually consider all cases
            if currLine.find("room-id") != -1:
                currLine = currLine.split(",")
                roomCorr.append(str(currLine[3]) + ", " + str(currLine[4]))
                # we need both room name and room id here
                roomList.append(roomCorr)
            # elif currLine.find("chilled/condensor water loop-id") != -1:
            #     currLine = currLine.split(",")
            #     roomCorr.append(currLine[2])
            #     roomList.append(roomCorr)
            # elif currLine.find("supply fan-id") != -1:
            #     currLine = currLine.split(",")
            #     roomCorr.append(currLine[4])
            #     roomList.append(roomCorr)
            # elif currLine.find("ahu-id") != -1:
            #     currLine = currLine.split(",")
            #     roomCorr.append(currLine[2])
            #     roomList.append(roomCorr)
            # elif currLine.find("hot water loop-id") != -1:
            #     currLine = currLine.split(",")
            #     roomCorr.append(currLine[2])
            #     roomList.append(roomCorr)
            # elif currLine.find("chiller-id") != -1:
            #     currLine = currLine.split(",")
            #     roomCorr.append(currLine[2])
            #     roomList.append(roomCorr)
            # elif currLine.find("exhaust fan-id") != -1:
            #     currLine = currLine.split(",")
            #     try:
            #         roomCorr.append(currLine[4])
            #     except IndexError:
            #         roomCorr.append(currLine[2])
            #     roomList.append(roomCorr)
            # elif currLine.find("condensor pump-id") != -1:
            #     currLine = currLine.split(",")
            #     roomCorr.append(currLine[2])
            #     roomList.append(roomCorr)
            else:
                pass

            i += 1
        # print(roomList)

        f.close()
    return roomList


# TODO: include config
# def read_colocation_data(config):
def read_colocation_data(building):

    x = []  # Store the value
    y = []  # Store the room number
    true_pos = []  # Store the filename
    cnt = 0  # Index for list y
    room_list = [] # Check if there is a sensor in the same room
    groundTruth = read_ground_truth(building)
    final_x, final_y, final_true_pos = [], [], []  # output

    # Path depends on where the method is called (where the main is)
    path = "rawdata/metadata/" + str(building)
    # print(path)
    folders = os.walk(path)

    for path, dir_list, files in folders:
        # print(path, dir_list, files)
        files.remove(".DS_Store")  # We don't want to read in .DS_Store file
        for filename in files:  # Iterating through each time series file
            # print(next_file)
            if filename.endswith("csv"):
                _, value = read_csv(os.path.join(path, filename))
                # print(value)
                # TODO: clean_temperature in Soda. What is sensor fore temperature?
                if filename == 'temperature.csv':
                    value = clean_temperature(value)

            # TODO: format of input for different buildings
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
                        # print(sensorID)

                # checking whether this sensor is in an existing room
                for sensor, tarSensorID, roomNumber in room_list:
                    # print(room, roomNumber)
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

    # Only want four-sensor rooms

    # Counting number of sensors in each room
    countDict = {}
    for index in y:
        a = countDict.get(index)
        if a is None:
            countDict.update({index: 1})
        else:
            countDict.update({index: a + 1})

    # Picking rooms with four sensors
    fourSensorRoom = []
    indexMap = {}
    roomNum = 0
    for key, value in countDict.items():
        if value == 4:
            # Intuitively we should add the key into the list
            # But the format requires the rooms to be indexes between 0 and length
            # So we create a mapping between real key and the number we want
            fourSensorRoom.append(key)
            indexMap.update({key: roomNum})
            roomNum += 1

    # Adding desired rooms into output list
    for i in range(len(y)):
        if y[i] in fourSensorRoom:
            # print(i, y[i], true_pos[i])
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
    x, y, true_pos = read_colocation_data(building)
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


def read_csv(path, config={}):
    f = open(path)
    timestamps, vals = [], []
    # print("-----")
    # print(path)
    for line in f.readlines():
        # if path == "../../rawdata/metadata/Soda/SODA1R376A_VAV.csv":
        #     print(line)
        if line == "":
            pass
        t, v = line.split(",")
        timestamps.append(int(t))
        vals.append(float(v))
    # return align_length(timestamps, vals, config.max_length)
    return align_length(timestamps, vals, 10000)


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


def split_colocation_train(x, y, test_index, split_method):
    train_x, train_y, test_x, test_y = [], [], [], []
    if split_method == 'room':
        for i in range(len(y)):
            if y[i] in test_index:
                test_x.append(x[i])
                test_y.append(y[i])
            else:
                train_x.append(x[i])
                train_y.append(y[i])
    else:
        for i in range(len(y)):
            if i not in test_index:
                train_x.append(x[i])
                train_y.append(y[i])
            else:
                test_y.append(i)
        test_x = x
    return train_x, train_y, test_x, test_y


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


# model-stn

class STN(nn.Module):
    def __init__(self, dropout_rate, input_channels):
        super(STN, self).__init__()
        self.d = dropout_rate

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=256, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=384, kernel_size=7, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=384, out_channels=128, kernel_size=6, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
        )
        self.dropout1 = nn.Dropout(self.d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        norm = x.norm(dim=1, p=2, keepdim=True)
        x = x.div(norm.expand_as(x))

        return x


# losses

class tripletLoss(nn.Module):
    def __init__(self, margin):
        super(tripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        distance_pos = (anchor - pos).pow(2).sum(1)
        distance_neg = (anchor - neg).pow(2).sum(1)
        loss = F.relu(distance_pos - distance_neg + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()


class combLoss(nn.Module):
    def __init__(self, margin, l=1):
        super(combLoss, self).__init__()
        self.margin = margin
        self.l = l

    def forward(self, anchor, pos, neg):
        distance_pos = (anchor - pos).pow(2).sum(1)
        distance_neg = (anchor - neg).pow(2).sum(1)
        distance_cen = (neg - anchor * 0.5 - pos * 0.5).pow(2).sum(1)
        loss = F.relu(distance_pos - self.l * distance_cen + self.margin)
        return loss.mean(), self.triplet_correct(distance_pos, distance_neg)

    def triplet_correct(self, d_pos, d_neg):
        return (d_pos < d_neg).sum()


# main

def parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-config', default='stn', type=str)
    parser.add_argument('-model', default='stn', type=str,
                        choices=['stn'])
    parser.add_argument('-loss', default='comb', type=str,
                        choices=['triplet', 'comb'])
    parser.add_argument('-seed', default=2, type=int,
                        help="Random seed")
    parser.add_argument('-log', default='stn', type=str,
                        help="Log directory")
    parser.add_argument('-facility', default=10606, type=int,
                        help="Log directory")
    parser.add_argument('-split', default='room', type=str,
                        help="split 1/5 sensors or rooms for test",
                        choices=['room', 'sensor'])
    args = parser.parse_args()
    # the file to be opened depends on where this method is called
    config = read_config('figs/' + args.config + '.yaml')
    return args, config

# Learning Helper
# def learn_buidling(x, y, log, config, args):
#     fold_recall = []
#     fold_room_acc = []
#
#     test_indexes = cross_validation_sample(50, 10)
#
#     log("test indexes:" + str(test_indexes) + "\n")
#
#     # train
#     # five folds, five sets of test_indexes
#     for fold, test_index in enumerate(test_indexes):
#
#         log("Now training fold: %d" % (fold))
#
#         # split training & testing
#         print("Test indexes: ", test_index)
#         train_x, train_y, test_x, test_y = split_colocation_train(x, y, test_index, args.split)
#         # important
#         # if y in test_index => get into test group
#         # else get into train group
#         # say test index = [14, 46, 48, 12, ...]
#         # then the 14th, 46th, 48th, ... room is test group
#         # corresponds to folder 456, 746, 752, etc
#
#         train_x = gen_colocation_triplet(train_x, train_y)
#         # This automatically uses y to get correct answer
#         # so we are able to identify right from wrong
#
#         total_triplets = len(train_x)
#         self.log("Total training triplets: %d\n" % (total_triplets))
#
#         train_loader = torch.utils.data.DataLoader(train_x, batch_size=self.config.batch_size, shuffle=True)
#         for epoch in range(self.config.epoch):
#
#             self.log("Now training %d epoch ......\n" % (epoch + 1))
#             total_triplet_correct = 0
#             for step, batch_x in enumerate(train_loader):
#                 # get into smaller groups
#                 if self.args.model == 'stn':
#                     # anchor = batch_x[0].cuda()
#                     # pos = batch_x[1].cuda()
#                     # neg = batch_x[2].cuda()
#
#                     anchor = batch_x[0]
#                     pos = batch_x[1]
#                     neg = batch_x[2]
#
#                 output_anchor = self.model(anchor)
#                 output_pos = self.model(pos)
#                 output_neg = self.model(neg)
#
#                 loss, triplet_correct = self.criterion(output_anchor, output_pos, output_neg)
#                 total_triplet_correct += triplet_correct.item()
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 if step % 200 == 0 and step != 0:
#                     self.log("loss " + str(loss) + "\n")
#                     self.log("triplet_acc " + str(triplet_correct.item() / self.config.batch_size) + "\n")
#
#             self.log("Triplet accuracy: %f" % (total_triplet_correct / total_triplets))

#         # TODO: How to calculate accuracy? How to predict?
#         solution, recall, room_wise_acc = self.predict(test_x, test_y, fold)
#         solution = solution.tolist()
#
#         self.log_result("fold: %d, epoch: %d\n" % (fold, epoch))
#         self.log_result("Acc: %f\n" % (recall))
#         self.log("fold: %d, epoch: %d\n" % (fold, epoch))
#         self.log("Acc: %f\n" % (recall))
#
#         for k in range(len(solution)):
#             for j in range(len(solution[k])):
#                 self.log_result(str(solution[k][j]) + ' ')
#             self.log_result('\n')
#         self.log_result('\n')
#
#     fold_recall.append(recall)
#     fold_room_acc.append(room_wise_acc)
#
# self.log("Final recall : %f \n" % (np.array(fold_recall).mean()))
# self.log("Final room accuracy : %f \n" % (np.array(fold_room_acc).mean()))
