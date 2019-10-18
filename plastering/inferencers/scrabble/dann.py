import numpy as np
import pdb
import random
from random import shuffle
import math
import shutil

from sklearn.manifold import TSNE
import arrow

import keras.backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import GradientReversal
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.callbacks import TensorBoard

import tensorflow as tf

from jasonhelper import *

def get_semi_random_subset(l, length):
    selected = []
    while True:
        shuffle(l)
        for item in l:
            selected.append(item)
            if len(selected) == length:
                yield selected
                selected = []

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

class DANN(object):
    def __init__(self,
                 data_dim,
                 nb_classes,
                 nb_domains,
                 batch_size=32,
                 mode='multilabel',
                 ):
        self.model = None
        self.net = None
        self.domain_invariant_features = None
        self.grl = None
        #self.opt = SGD()
        #self.opt = 'sgd'
        self.opt = 'rmsprop'
        #self.opt = RMSprop(lr=0.0001)
        self.data_dim = data_dim
        self.nb_classes = nb_classes
        self.nb_domains = nb_domains
        self.batch_size = batch_size
        self.mode = mode
        if self.mode == 'multilabel':
            self.classifier_loss = 'binary_crossentropy'
            self.classifier_activation = 'sigmoid'
        elif self.mode == 'multiclass':
            self.classifier_loss = 'categorical_crossentropy'
            self.classifier_activation = 'softmax'
        else:
            raise Exception('not implemented')
        self.tb_log_dir_l = 'logs/l'
        self.tb_log_dir_d = 'logs/d'
        shutil.rmtree(self.tb_log_dir_l)
        shutil.rmtree(self.tb_log_dir_d)
        os.mkdir(self.tb_log_dir_l)
        os.mkdir(self.tb_log_dir_d)
        print('loss function: {0}'.format(self.classifier_loss))
        print('activation: {0}'.format(self.classifier_activation))
        self.d_lambda = 1


    def build_all(self):
        pass

    def _build_feature_extractor(self, model_input):
        '''Build segment of net for feature extraction.'''
        net = Dense(128, activation='relu')(model_input)
        net = Dropout(0.1)(net)
        self.domain_invariant_features = net
        return net

    def _build_label_classifier(self, model_input):
        net = model_input
        net = Dense(64, activation='relu')(net)
        #net = Dropout(0.1)(net)
        net = Dense(self.nb_classes,
                    activation=self.classifier_activation,
                    name='classifier_output',
                    )(net)
        return net

    def _build_domain_classifier(self, model_input, hp_lambda):
        net = model_input
        self.grl = GradientReversal(hp_lambda)
        net = self.grl(net)
        net = Dense(32,
                    activation='relu',
                    #kernel_regularizer=regularizers.l2(0.01),
                    )(net)
        net = Dropout(0.1)(net)
        net = Dense(self.nb_domains,
                       activation='softmax',
                       name='domain_output')(net)
        return net

    def build_source_model(self, main_input):
        net = self._build_feature_extractor(main_input)
        net = self._build_label_classifier(net)
        model = Model(input=main_input, output=net)
        model.compile(loss={'classifier_output': self.classifier_loss},
                      optimizer=self.opt, metrics=['accuracy'])
        return model

    def build_input(self):
        return Input(shape=(self.data_dim,), name='main_input')

    #def naive_fit(self, x_s, y_s, nb_epochs=300):
    def naive_fit(self, x_s, y_s, domain_s, x_t, domain_t, target_domain_index, nb_epochs=300):
        main_input = self.build_input()
        self.naive_model = self.build_source_model(main_input)
        self.naive_model.fit(x_s, y_s, batch_size=self.batch_size, epochs=nb_epochs, verbose=False)

    def naive_predict(self, data):
        res = self.naive_model.predict(data)
        return res

    def build_dann_model(self, hp_lambda=1.0):
        main_input = self.build_input()
        net = self._build_feature_extractor(main_input)
        domain_out = self._build_domain_classifier(net, hp_lambda)
        label_out = self._build_label_classifier(net)

        model_label = Model(input=main_input, output=label_out)
        model_domain = Model(input=main_input, output=domain_out)

        model_label.compile(loss=self.classifier_loss,
                            optimizer=self.opt,
                            metrics=['accuracy'],
                            )

        model_domain.compile(loss='categorical_crossentropy',
                             optimizer=self.opt,
                             metrics=['accuracy'],
                             )
        return (model_label, model_domain)

    def build_tsne_model(self, main_input):
        '''Create model to output intermediate layer
        activations to visualize domain invariant features'''
        tsne_model = Model(input=main_input,
                           output=self.domain_invariant_features)
        return tsne_model

    def batch_gen_no_matching(self, data_s, labels_s, domains_s, data_t, domains_t, target_domain_index):
        ids = ['s:{0}'.format(idx) for idx in range(data_s.shape[0])] + \
            ['t:{0}'.format(idx) for idx in range(data_t.shape[0])]
        shuffle(ids)
        for batch_ids in striding_windows(ids, self.batch_size):
            s_batch_ids = [int(idx.split(':')[-1]) for idx in batch_ids if idx.split(':')[0] == 's']
            t_batch_ids = [int(idx.split(':')[-1]) for idx in batch_ids if idx.split(':')[0] == 't']
            t_data = (data_t[t_batch_ids], domains_t[t_batch_ids])
            s_data = (data_s[s_batch_ids], domains_s[s_batch_ids], labels_s[s_batch_ids])
            yield s_data, t_data

    def batch_gen(self, data_s, labels_s, domains_s, data_t, domains_t, target_domain_index):
        """
        Note: Assume that data_t is from a single target domain
        A batch consists of multiple "single batch"es. Each "single batch" is fro a single domain.
        """
        domain_ids_list = []
        ids_list = []
        for domain_idx, domain in enumerate(domains_s.T):
            domain = domain.T
            ids = ['s:{0}'.format(idx) for idx in np.where(domain==1)[0]]
            #s_ids = ['{1}'.format(domain_idx, s_id) for s_id in np.where(domain==1)[0]]
            shuffle(ids)
            ids_list.append(ids)
        ids_list[target_domain_index] += ['t:{0}'.format(idx) for idx in range(domains_t.shape[0])]
        domains_nbs = [len(s_ids) for s_ids in ids_list]
        max_domains_nb = max(domains_nbs)
        single_batch_size = int(self.batch_size / domains_s.shape[1])
        ids_generators = [get_semi_random_subset(ids, single_batch_size) for ids in ids_list]
        for i in range(math.ceil(max_domains_nb / single_batch_size)):
            batch_ids = []
            for ids_generator in ids_generators:
                batch_ids += next(ids_generator)
            s_batch_ids = [int(idx.split(':')[-1]) for idx in batch_ids if idx.split(':')[0] == 's']
            t_batch_ids = [int(idx.split(':')[-1]) for idx in batch_ids if idx.split(':')[0] == 't']
            t_data = (data_t[t_batch_ids], domains_t[t_batch_ids])
            s_data = (data_s[s_batch_ids], domains_s[s_batch_ids], labels_s[s_batch_ids])
            yield s_data, t_data

    def predict(self, data):
        res = self.model_label.predict(data)
        pdb.set_trace()
        return res

    def fit(self, x_s, y_s, domain_s, x_t, domain_t, target_domain_index, nb_epochs=300):
        assert x_s.shape[0] == y_s.shape[0]
        assert domain_s.shape[0] == y_s.shape[0]
        assert x_t.shape[0] == domain_t.shape[0]
        batches_per_epoch = x_s.shape[0] / self.batch_size
        num_steps = nb_epochs * batches_per_epoch
        self.model_label, self.model_domain = self.build_dann_model(hp_lambda=1)
        tb_l = TensorBoard(log_dir=self.tb_log_dir_l)
        tb_l.set_model(self.model_label)
        tb_d = TensorBoard(log_dir=self.tb_log_dir_d)
        tb_d.set_model(self.model_domain)

        for i in range(0, nb_epochs):
            start_time = arrow.get()
            batches = self.batch_gen_no_matching(x_s, y_s, domain_s, x_t, domain_t, target_domain_index)
            #batches = self.batch_gen(x_s, y_s, domain_s, x_t, domain_t, target_domain_index)
            metrics_l_list = []
            weights_l = []
            metrics_d_list = []
            weights_d = []
            batch_weights = []
            for j, ((x_s_batch, d_s_batch, y_s_batch), (x_t_batch, d_t_batch)) in enumerate(batches):
                s_size = x_s_batch.shape[0]
                t_size = x_t_batch.shape[0]
                if not s_size or not t_size:
                    continue
                domain_nbs = np.sum(d_s_batch, axis=0) + np.sum(d_t_batch, axis=0)
                if 0 in domain_nbs:
                    continue

                domain_rates = np.divide(1, domain_nbs)
                domain_rates /= np.sum(domain_rates)
                domain_x = np.vstack([x_s_batch, x_t_batch])
                domain_y = np.vstack([d_s_batch, d_t_batch])
                #domain_weights = np.array(np.matmul(domain_y, domain_rates.T).T.tolist()[0])
                domain_weights = np.array(np.matmul(domain_y, domain_rates.T))
                s_nb = x_s_batch.shape[0]
                domain_weights = domain_weights / np.sum(domain_weights) * s_nb * self.d_lambda
                if len(domain_weights.shape) == 2:
                    domain_weights = domain_weights.reshape((domain_weights.shape[0]))

                metrics_l = self.model_label.train_on_batch(x_s_batch, y_s_batch)
                metrics_l_list.append(metrics_l)
                metrics_d = self.model_domain.train_on_batch(domain_x, domain_y, sample_weight=domain_weights)
                metrics_d_list.append(metrics_d)
                batch_weights.append(s_nb)
            averaged_metrics_l = np.average(np.array(metrics_l_list), axis=0, weights=batch_weights)
            averaged_metrics_d = np.average(np.array(metrics_d_list), axis=0, weights=batch_weights)
            print(self.model_label.metrics_names, averaged_metrics_l)
            print(self.model_domain.metrics_names, averaged_metrics_d)
            write_log(tb_l, self.model_label.metrics_names, averaged_metrics_l, i)
            write_log(tb_d, self.model_domain.metrics_names, averaged_metrics_d, i)
            end_time = arrow.get()
            print('{0} epoch took {1}'.format(i, end_time - start_time))

