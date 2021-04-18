import time

import torch
from plastering.inferencers import Inferencer
from .relational_inference_helper import *
from plastering.inferencers.algorithm.GeneticAlgorithm.colocation import run
import scipy.io as scio


@Inferencer()
class RelationalInference(object):

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
        self.target_x = {}
        self.target_y = {}
        self.target_true_pos = {}
        self.sensor_count = config.sensor_count
        self.target_building = target_building

        self.log(str(time.asctime(time.localtime(time.time()))))

        # initialize the model depending on the configuration
        if self.args.loss == 'triplet':
            self.criterion = tripletLoss(margin=1).cuda()
            # self.criterion = tripletLoss(margin=1)
        elif self.args.loss == 'comb':
            self.criterion = combLoss(margin=1).cuda()
            # self.criterion = combLoss(margin=1)

        if self.args.model == 'stn':
            self.model = STN(self.config.dropout, 2 * self.config.k_coefficient).cuda()
            # self.model = STN(self.config.dropout, 2 * self.config.k_coefficient)

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

        for source_building in source_buildings:
            self.x, self.y, self.true_pos = read_in_data(source_building, self.config)
            self.log("%d total sensors, %d frequency coefficients, %d windows\n" % (
                len(self.x), self.x[0].shape[0], self.x[0].shape[1]))
            self.learn_auto()

    def learn_auto(self):

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

    def update_model(self):
        # currently not supported
        pass

    def predict(self):
        # This method tests the target building using the model after training the source buildings.
        self.target_x, self.target_y, self.target_true_pos = read_in_data(self.target_building, self.config)
        return self.test_colocation(self, self.target_x, self.target_y, self.config.fold)

    def select_informative_samples(self):
        # currently not supported
        pass

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
        self.log("Edge-wise accuracy: %f \n" % (acc))

        self.model.train()
        return best_solution, recall, room_wise_acc
