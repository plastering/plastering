import time
import torch

from . import Inferencer
from .algorithm.GeneticAlgorithm.colocation import run

from .relational_inference.relational_inference_helper import *
from .relational_inference.Data import *
from .relational_inference.loss import tripletLoss, combLoss
from .relational_inference.stn import STN
from .relational_inference.util import cal_room_acc

import scipy.io as scio


@Inferencer()
class RelationalInference(object):
    """
    # input parameters
    - target_building (str): name of the target building. this can be arbitrary later
    - source_buildings (list(str)): list of buildings already known.
    - conf: dictionary of other configuration parameters.
    """

    def __init__(self,
                 target_building,
                 target_srcids,
                 source_buildings,
                 config={},
                 args={},
                 **kwargs
                 ):

        # set up fields
        self.args = args
        self.config = config
        self.log, self.log_result, self.log_path = set_up_logging(config, args)
        self.criterion = {}
        self.model = {}
        self.optimizer = {}
        self.loss = {}
        self.target_building = target_building

        self.log(str(time.asctime(time.localtime(time.time()))))

        # initialize the model depending on the configuration
        if self.args.loss == 'triplet':
            self.criterion = tripletLoss(margin=1).cuda()
            # self.criterion = tripletLoss(margin=1)
        elif self.args.loss == 'comb':
            self.criterion = combLoss(margin=1).cuda()
            # self.criterion = combLoss(margin=1)
        elif self.args.loss == 'angular':
            self.criterion = self.angularLoss(margin=1).cuda()
        elif self.args.loss == 'softmax':
            self.criterion = self.softmaxtripletLoss().cuda()

        if self.args.model == 'stn':
            self.model = STN(self.config.dropout, 2 * self.config.k_coefficient).cuda()
            # self.model = STN(self.config.dropout, 2 * self.config.k_coefficient)
        elif self.args.model == 'han':
            self.model = STN(self.config.dropout, 2 * self.config.k_coefficient).cuda()

        if self.config.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9,
                                             weight_decay=self.config.weight_decay)
        elif self.config.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                              weight_decay=self.config.weight_decay)

        if self.config.grad_norm > 0:
            nn.utils.clip_grad_value_(self.model.parameters(), self.config.grad_norm)
            for p in self.model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -self.config.grad_norm, self.config.grad_norm))

        print("Model : ", self.model)
        print("Criterion : ", self.criterion)

        # co-location
        # notice this if statement is quite arbitrary; it depends on the yaml file.
        if args.config == "colocation":
            self.target_x = {}
            self.target_y = {}
            self.target_true_pos = {}
            self.sensor_count = config.sensor_count
            for source_building in source_buildings:
                self.x, self.y, self.true_pos = read_in_data(source_building, self.config)
                self.log("%d total sensors, %d frequency coefficients, %d windows\n" % (
                    len(self.x), self.x[0].shape[0], self.x[0].shape[1]))
                self.learn_auto()

        # co-equipment
        if args.config == "coequipment":
            self.learn_auto()

    def learn_auto(self):
        if self.args.config == "colocation":
            self.train_colocation()
        elif self.args.config == "coequipment":
            self.train_coequipment(self.target_building, self.source_buildings)

    def update_model(self):
        # currently not supported
        pass

    def predict(self):
        # This method tests the target building using the model after training the source buildings.

        # co-location
        if self.args.config == "colocation":
            self.target_x, self.target_y, self.target_true_pos = read_in_data(self.target_building, self.config)
            return self.test_colocation(self, self.target_x, self.target_y, self.config.fold)

        # co-equipment
        if self.args.config == "coequipment":
            # print(read_coequipment_data(self.config, self.args, self.target_building, self.source_buildings))
            ahu_x, ahu_y, vav_x, vav_y, test_indices, mapping = \
                read_coequipment_data(self.config, self.args, self.target_building, self.source_buildings)
            return self.test_coequipment(ahu_x, ahu_y, vav_x, vav_y, mapping, self.target_building)

    def select_informative_samples(self):
        # currently not supported
        pass

    # helper methods

    def train_colocation(self):
        fold_recall = []
        fold_room_acc = []

        # initialize the test indexes
        test_indexes = cross_validation_sample(30, 10)

        self.log("test indexes:" + str(test_indexes) + "\n")

        # train
        # sets of test_indexes
        for fold, test_index in enumerate(test_indexes):

            self.log("Now training fold: %d" % fold)

            # split training & testing
            print("Test indexes: ", test_index)
            train_x, train_y, train_true_pos, test_x, test_y, test_true_pos = \
                split_colocation_train(self.x, self.y, self.true_pos, test_index, self.args.split)
            # if y in test_index => get into test group
            # else get into train group
            # say test index = [14, 46, 48, 12, ...]
            # then the 14th, 46th, 48th, ... sensors are the test group

            train_x = gen_colocation_triplet(train_x, train_y)
            # generates triplets with anchor, pos, and neg

            total_triplets = len(train_x)
            self.log("Total training triplets: %d\n" % total_triplets)

            train_loader = torch.utils.data.DataLoader(train_x, batch_size=self.config.batch_size, shuffle=True)

            for epoch in range(self.config.epoch):

                self.log("Now training %d epoch ......\n" % (epoch + 1))
                total_triplet_correct = 0

                # each batch is a smaller group of the training group
                for step, batch_x in enumerate(train_loader):
                    if self.args.model == 'stn':
                        anchor = batch_x[0].cuda()
                        pos = batch_x[1].cuda()
                        neg = batch_x[2].cuda()

                    output_anchor = self.model(anchor)
                    output_pos = self.model(pos)
                    output_neg = self.model(neg)

                    # evaluate the loss
                    self.loss, triplet_correct = self.criterion(output_anchor, output_pos, output_neg)
                    total_triplet_correct += triplet_correct.item()

                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()

                    if step % 200 == 0 and step != 0:
                        self.log("loss " + str(self.loss) + "\n")
                        self.log("triplet_acc " + str(triplet_correct.item() / self.config.batch_size) + "\n")

                self.log("Triplet accuracy: %f" % (total_triplet_correct / total_triplets))

                solution, recall, room_wise_acc = self.test_colocation(test_x, test_y, fold)
                solution = solution.tolist()

                self.log_result("fold: %d, epoch: %d\n" % (fold, epoch))
                self.log_result("Acc: %f\n" % (recall))
                self.log("fold: %d, epoch: %d\n" % (fold, epoch))
                self.log("Acc: %f\n" % (recall))

                for k in range(len(solution)):
                    for j in range(len(solution[k])):
                        self.log_result(str(solution[k][j]) + ' ')
                    self.log_result('\n')
                self.log_result('\n')

            fold_recall.append(recall)
            fold_room_acc.append(room_wise_acc)

        self.log("Final recall : %f \n" % (np.array(fold_recall).mean()))
        self.log("Final room accuracy : %f \n" % (np.array(fold_room_acc).mean()))

    def test_colocation(self, test_x, test_y, fold):
        self.model.eval()

        with torch.no_grad():
            if self.args.model == 'stn':
                out = self.model(torch.from_numpy(np.array(test_x)).cuda())
                # model(tensor(3D array))
                # Array of 2D arrays
                # each 2D array is the STFT output
            test_triplet = gen_colocation_triplet(test_x, test_y, prevent_same_type=True)
            # [anchor, pos, neg]
            test_loader = torch.utils.data.DataLoader(test_triplet, batch_size=1, shuffle=False)
            cnt = 0
            for step, batch_x in enumerate(test_loader):
                if self.args.model == 'stn':
                    anchor = batch_x[0].cuda()
                    pos = batch_x[1].cuda()
                    neg = batch_x[2].cuda()

                output_anchor = self.model(anchor)
                output_pos = self.model(pos)
                output_neg = self.model(neg)
                # anchor, pos, neg after training

                # cnt counts the correct triplets
                distance_pos = (output_anchor - output_pos).pow(2).sum(1).pow(1 / 2)
                distance_neg = (output_anchor - output_neg).pow(2).sum(1).pow(1 / 2)
                if distance_neg > distance_pos:
                    cnt += 1

            self.log("Testing triplet acc: %f" % (cnt / len(test_triplet)))

        test_out = out.cpu().tolist()
        test_corr = np.corrcoef(np.array(test_out))

        # save the correlation matrix
        scio.savemat('./result/RelationalInferenceOutput/corr_' + str(fold) + '.mat', {'corr': test_corr})

        # calling the genetic algorithm to get the final result
        best_solution, acc, ground_truth_fitness, best_fitness = run.ga(path_m='./result/RelationalInferenceOutput'
                                                                               '/corr_' + str(fold) + '.mat',
                                                                        path_c='./figs/10_rooms.json')

        # calculate the accuracy
        recall, room_wise_acc = cal_room_acc(best_solution, self.sensor_count)
        # best_solution [[sensor1, sensor2, sensor3, sensor4],[ ... ], ...]
        # why is best_solution from 0 to 40? 10 rooms. 4 sensors each room
        # Group sensors together
        self.log("recall = %f, room_wise_acc = %f:\n" % (recall, room_wise_acc))

        self.log("Ground Truth Fitness %f Best Fitness: %f \n" % (ground_truth_fitness, best_fitness))
        self.log("Edge-wise accuracy: %f \n" % acc)

        self.model.train()
        return best_solution, recall, room_wise_acc

    def train_coequipment(self, test, train):
        ahu_x, ahu_y, vav_x, vav_y, test_indices, mapping = read_coequipment_data(self.config, self.args, test, train)

        # print(test_indices)
        epochs_acc = []
        total_wrongs = dict()
        for fold, test_index in enumerate(test_indices):
            epochs_acc.append([])
            self.log("Now training fold: %d" % (fold))
            train_vav_x, train_vav_y, test_vav_x, test_vav_y = split_coequipment_train(vav_x, vav_y, test_index, train,
                                                                                       test)
            train_x = gen_coequipment_triplet(ahu_x, ahu_y, train_vav_x, train_vav_y, mapping)
            test_y = test_vav_y
            total_triplets = len(train_x)
            self.log("Total training triplets: %d\n" % (total_triplets))
            testahu = dict()
            for v in test_y:
                if mapping[v[0]][v[1]] not in testahu:
                    testahu[mapping[v[0]][v[1]]] = 1
                else:
                    testahu[mapping[v[0]][v[1]]] += 1

            # model = torch.load(log_path + 'model.pkl')
            print("testahus :\n", testahu)
            for epoch in range(self.config.epoch):
                # print(train_x)
                train_loader = torch.utils.data.DataLoader(train_x, batch_size=self.config.batch_size, shuffle=True)
                self.log("Now training %d epoch ......\n" % (epoch + 1))
                total_triplet_correct = 0
                for step, batch_x in enumerate(train_loader):

                    anchor = batch_x[0].cuda()
                    pos = batch_x[1].cuda()
                    neg = batch_x[2].cuda()

                    output_anchor = self.model(anchor)
                    output_pos = self.model(pos)
                    output_neg = self.model(neg)

                    loss, triplet_correct = self.criterion(output_anchor, output_pos, output_neg)
                    total_triplet_correct += triplet_correct.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if step % 200 == 0 and step != 0:
                        self.log("loss " + str(loss) + "\n")
                        self.log("triplet_acc " + str(triplet_correct.item() / self.config.batch_size) + "\n")

                self.log("Triplet accuracy: %f" % (total_triplet_correct / total_triplets))

                # torch.save(model, log_path + args.task + '_' + args.model + '_model.pkl')
                acc, wrongs = self.test_coequipment(ahu_x, ahu_y, test_vav_x, test_vav_y, mapping, test)
                epochs_acc[fold].append(acc)
                for keys in wrongs:
                    if keys in total_wrongs:
                        total_wrongs[keys] += wrongs[keys]
                    else:
                        total_wrongs[keys] = wrongs[keys]
                print("Wrong equipments : ")
                print(total_wrongs)

        overall_epoch_acc = [0 for i in range(self.config.epoch)]

        for i in range(len(epochs_acc)):
            for j in range(len(epochs_acc[0])):
                overall_epoch_acc[j] += epochs_acc[i][j] / len(epochs_acc)

        # print(overall_epoch_acc)
        self.log("Best accuracy : %f, best epoch: %d\n" % (
            max(overall_epoch_acc), overall_epoch_acc.index(max(overall_epoch_acc))))
        return max(overall_epoch_acc)

    def test_coequipment(self, ahu_x, ahu_y, vav_x, vav_y, mapping, test):
        # TODO: how to judge if correct/wrong?
        #  in SODA, simply compare the 4 and 5 th character
        #  i.e. substring(3,5)
        #  but how to incorporate that into this method?
        self.model.eval()
        wrongs = dict()
        facilities = []
        for ahu in ahu_y:
            if ahu[0] not in facilities:
                facilities.append(ahu[0])
        acc = []

        for f_id in facilities:
            if f_id != test:
                continue
            mapping_fid = mapping[f_id]
            test_vav = []
            test_ahu = []
            test_ahu_y = []
            test_vav_y = []
            for i in range(len(ahu_y)):
                if ahu_y[i][0] == f_id:
                    test_ahu.append(ahu_x[i])
                    test_ahu_y.append(ahu_y[i][1])

            for i in range(len(vav_y)):
                if vav_y[i][0] == f_id:
                    test_vav.append(vav_x[i])
                    test_vav_y.append(vav_y[i][1])

            with torch.no_grad():
                vav_out = self.model(torch.from_numpy(np.array(test_vav)).cuda())
                ahu_out = self.model(torch.from_numpy(np.array(test_ahu)).cuda())

            vav_out = vav_out.cpu().tolist()
            ahu_out = ahu_out.cpu().tolist()
            total_pairs = [0 for i in range(len(ahu_out))]
            repeate = [0 for i in range(len(vav_out))]
            for i in range(len(vav_out)):
                total_pairs[test_ahu_y.index(mapping_fid[test_vav_y[i]])] += 1
            # print(total_pairs)
            total = len(vav_out)
            cnt = 0
            euclidean_norm = lambda x, y: np.abs(x - y)
            for i, vav_emb in enumerate(vav_out):
                min_dist = 0xffff
                min_idx = 0
                for j, ahu_emb in enumerate(ahu_out):
                    dist = np.linalg.norm(np.array(ahu_emb) - np.array(vav_emb))
                    if dist < min_dist:  # and total_pairs[j] > 0:
                        min_dist = dist
                        min_idx = j
                total_pairs[min_idx] -= 1
                min_dist = 0xfff
                if mapping_fid[test_vav_y[i]] == test_ahu_y[min_idx]:
                    cnt += 1
                else:
                    if mapping_fid[test_vav_y[i]] in wrongs:
                        wrongs[mapping_fid[test_vav_y[i]]] += 1
                    else:
                        wrongs[mapping_fid[test_vav_y[i]]] = 1
            self.log("Fid: %s Acc: %f\n" % (f_id, cnt / total))
            acc.append(cnt / total)
        # print(wrongs)
        self.model.train()
        return acc[0], wrongs
