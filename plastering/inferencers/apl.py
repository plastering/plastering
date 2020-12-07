import sys
import json
import pdb
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count, combinations
import pickle
import scipy
from scipy.special import softmax
import sys
from sklearn.feature_extraction.text import CountVectorizer as CV
import re
import copy
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn_crfsuite import CRF
import nltk
import argparse

import pandas as pd
import numpy as np
from . import Inferencer

from plastering.metadata_interface import *
from plastering.common import *
from plastering.rdf_wrapper import get_top_class


class BuildDataLoader:
    
    def __init__(self, folder):
        self.sequence = []
        self.word_dict = {}
        self.label_dict = {}
        self.folder = folder
        # Load Labeled Metadata
        with open('groundtruth/{0}_labeled_metadata.json'.format(self.folder), 'r') as fp:
            data = json.load(fp)

        for srcid, doc in data.items():
            if(doc['fullparsing'].get('BACnetName') == None or len(doc['fullparsing'].get('BACnetName')) == 0):
                continue
            current_data = doc['fullparsing'].get('BACnetName')
            x = []
            y = []
            for i in range(len(current_data)):
                if(current_data[i][0].isdigit()):
                    current_data[i][0] = 'NUM'
                else:
                    current_data[i][0] = current_data[i][0].lower().replace("\n",'')
                x.append(current_data[i][0])
                y.append(current_data[i][1])
            
            self.sequence.append((x,y))
        print(len(self.sequence))
    def shuffle(self, seed = 4):
        random.Random(4).shuffle(self.sequence)

# class ActivePartialLabelling(Inferencer):
class ActivePartialLabelling(Inferencer):
    def __init__(self, source, budget, method, strategy, window, cflag, testM, beta, tune):

        SOURCE = source
        # Arguments might have to be changed here
        DATA_PATH = SOURCE
        self.PRETRAIN_SIZE = 15
        self.CANDIDATE_SIZE = 600
        self.VALIDATE_SIZE = 200
        self.TEST_SIZE = 200
        self.BUDGET = budget

        #inductive or transductive labeling
        self.M = testM
        self.BETA = beta
        self.METHOD = method #choice: none, selfSim, testSim
        #fully or partial labeling
        self.SUBSEQ_FLAG = True if cflag == 1 else False
        self.SUBSEQ_SIZE = window
        self.STRATEGY = strategy #choice: fully, partial
        self.NORM = True if tune == 'norm' else False

        # Load Data, could be a different method possibly
        data = BuildDataLoader(DATA_PATH)
        data.shuffle(8)
        self.pretrain_list = data.sequence[:self.PRETRAIN_SIZE]
        self.test_list = data.sequence[-self.TEST_SIZE:]
        self.validation_list = data.sequence[-self.TEST_SIZE - self.VALIDATE_SIZE : self.TEST_SIZE]
        self.candidate_list  = data.sequence[self.PRETRAIN_SIZE : self.PRETRAIN_SIZE + self.CANDIDATE_SIZE]

        # create a crf model that we will use
        self.crf = CrfModel(data)
        self.crf.add_instances(self.pretrain_list)
        self.crf.train()

        self.count = sum([len(seq[1]) for seq in self.pretrain_list]) 
        self.cost_list = [self.count]

        (in_acc, out_acc, all_acc) = self.crf.evaluate_acc(self.test_list)
        self.in_acc_list = [in_acc]
        self.out_acc_list = [out_acc]
        self.all_acc_list = [all_acc]

        # precompute: vectorized and clustered test set.
        Xs = [seq[0] for seq in self.validation_list]
        Xs.extend([seq[0] for seq in self.candidate_list])
        vec, _ = string_vectorize(Xs)
        validation_vec = vec[:len(self.validation_list)].tolist()
        candidate_vec = vec[len(self.validation_list):].tolist()

        # Pre-calculate similarity: both between validation-test and validation-validate
        sim_matrix_test = np.zeros((len(candidate_vec), len(validation_vec)))
        sim_matrix_self = np.zeros((len(candidate_vec), len(candidate_vec)))

        iterator = tqdm(range(len(candidate_vec)))
        iterator = tqdm(range(len(candidate_vec)))
        for i in iterator:
            for j in range(len(validation_vec)):
                sim_matrix_test[i, j] = 1 - scipy.spatial.distance.cosine(candidate_vec[i], validation_vec[j]) # cosine distance is 1-cosine(a,b)
            for j in range(len(candidate_vec)):
                sim_matrix_self[i, j] = 1 - scipy.spatial.distance.cosine(candidate_vec[i], candidate_vec[j])
        iterator.close()
        print ('Similarity done!')
    #second constructor
    def __init__(self,
                 target_building,
                 target_srcids,
                 source_buildings=[],
                 config={},
                 **kwargs,
                 ):
        parser = argparse.ArgumentParser(description='sequence active learning')
        parser.add_argument('-s', '--source',  help='data source')
        parser.add_argument('-n', '--budget', type=int, default=1000)
        parser.add_argument('-f', '--method', choices=['none', 'selfSim', 'testSim'], help='inductive or transductive labeling', default='none')
        parser.add_argument('-l', '--strategy', choices=['fully', 'partial'], help='fully or partial labeling', default="fully")
        parser.add_argument('-w', '--window', type=int,  help='window size for subsequence', default=8)
        parser.add_argument('-c', '--cflag', type=int, help='continuous subsequence or not', default=0)
        parser.add_argument('-m', '--testM', type=int, help='top m test instances', default="50")
        parser.add_argument('-b', '--beta',  type=float, help='beta of information density', default=1.0)
        parser.add_argument('-t', '--tune', choices=['none', 'norm'], help='normalize the metric or not', default="none")
        
        args = parser.parse_args()

        SOURCE = args.source
        # Arguments might have to be changed here
        DATA_PATH = SOURCE
        self.PRETRAIN_SIZE = 15
        self.CANDIDATE_SIZE = 600
        self.VALIDATE_SIZE = 200
        self.TEST_SIZE = 200
        self.BUDGET = args.budget

        #inductive or transductive labeling
        self.M = args.testM
        self.BETA = args.beta
        self.METHOD = args.method #choice: none, selfSim, testSim
        #fully or partial labeling
        self.SUBSEQ_FLAG = True if args.cflag == 1 else False
        self.SUBSEQ_SIZE = args.window
        self.STRATEGY = args.strategy #choice: fully, partial
        self.NORM = True if args.tune == 'norm' else False

        # Load Data, could be a different method possibly
        data = BuildDataLoader(DATA_PATH)
        data.shuffle(8)
        self.pretrain_list = data.sequence[:self.PRETRAIN_SIZE]
        self.test_list = data.sequence[-self.TEST_SIZE:]
        self.validation_list = data.sequence[-self.TEST_SIZE - self.VALIDATE_SIZE : self.TEST_SIZE]
        self.candidate_list  = data.sequence[self.PRETRAIN_SIZE : self.PRETRAIN_SIZE + self.CANDIDATE_SIZE]

        # create a crf model that we will use
        self.crf = CrfModel(data)
        self.crf.add_instances(self.pretrain_list)
        self.crf.train()

        self.count = sum([len(seq[1]) for seq in self.pretrain_list]) 
        self.cost_list = [self.count]

        (in_acc, out_acc, all_acc) = self.crf.evaluate_acc(self.test_list)
        self.in_acc_list = [in_acc]
        self.out_acc_list = [out_acc]
        self.all_acc_list = [all_acc]

        # precompute: vectorized and clustered test set.
        Xs = [seq[0] for seq in self.validation_list]
        Xs.extend([seq[0] for seq in self.candidate_list])
        vec, _ = string_vectorize(Xs)
        validation_vec = vec[:len(self.validation_list)].tolist()
        candidate_vec = vec[len(self.validation_list):].tolist()

        # Pre-calculate similarity: both between validation-test and validation-validate
        sim_matrix_test = np.zeros((len(candidate_vec), len(validation_vec)))
        sim_matrix_self = np.zeros((len(candidate_vec), len(candidate_vec)))

        iterator = tqdm(range(len(candidate_vec)))
        iterator = tqdm(range(len(candidate_vec)))
        for i in iterator:
            for j in range(len(validation_vec)):
                sim_matrix_test[i, j] = 1 - scipy.spatial.distance.cosine(candidate_vec[i], validation_vec[j]) # cosine distance is 1-cosine(a,b)
            for j in range(len(candidate_vec)):
                sim_matrix_self[i, j] = 1 - scipy.spatial.distance.cosine(candidate_vec[i], candidate_vec[j])
        iterator.close()
        print ('Similarity done!')
    def load_data(self, folder,**kwargs):
        return

    def learn_auto(self, **kwargs):
        visited_candidate_idx = []
        iterator = tqdm(range(self.CANDIDATE_SIZE))
        for seqs_size in iterator:
            if self.cost_list[-1] > self.BUDGET:
                break
            
            # Sort the test set based on confidence.
            prob_test_list = []
            for i in range(len(self.validation_list)):
                (prob_per_token, _, prob_sum) = self.crf.compute_confidence(self.validation_list[i])
                prob_test_list.append(prob_sum)
            rank_idx_test = np.argsort(np.array(prob_test_list), kind='mergesort').tolist()[::-1]

            # Calculate the average similarity between the unlabeled samples and the selected test samples.
            distance = []
            if self.METHOD != 'none':
                if self.METHOD == 'testSim':
                    distance = np.sum(sim_matrix_test[:, rank_idx_test[:M]], axis=1) / M
                else:
                    distance = np.sum(sim_matrix_self, axis=1) / (len(candidate_vec)-1)
                # mean_dist = np.mean(distance)
                # std_dist = np.std(distance)
                # distance = [(distance[i] - mean_dist) / std_dist for i in range(len(candidate_list))]


            ####
            # Compute the top-K tokens and its seq_idx: subsequence with or without SEBSEQ_FLAG
            prob_list = []
            subseq_idx_list = []
            for i in range(len(self.candidate_list)):
                (prob_per_token, prob_sum) = self.crf.compute_entropy(self.candidate_list[i])
                prob_sum /= len(self.candidate_list[i][1])
                if self.STRATEGY == 'partial':
                    subseq_idxs = []
                    subseq_prob_sum = -sys.maxsize
                    if self.SUBSEQ_FLAG:
                        end_p = len(prob_per_token) - self.SUBSEQ_SIZE + 1
                        for k in range(0, end_p): # the largest subsequence
                            prob_tmp = sum([prob_per_token[k+j] for j in range(self.SUBSEQ_SIZE)]) / self.SUBSEQ_SIZE
                            if prob_tmp > subseq_prob_sum:
                                subseq_prob_sum = prob_tmp
                                subseq_idxs = [k+j for j in range(self.SUBSEQ_SIZE)]
                        if end_p < 1: # if length is not longer than subseq_size
                            subseq_prob_sum = prob_sum / len(prob_per_token)
                            subseq_idxs = range(0, len(prob_per_token))
                    else:
                        token_sorted = np.argsort(np.array(prob_per_token), kind='mergesort').tolist()[::-1]
                        subseq_idxs = [token_sorted[k] for k in range(min(self.SUBSEQ_SIZE, len(prob_per_token)))]
                        subseq_prob_sum = sum([prob_per_token[k] for k in subseq_idxs]) / len(subseq_idxs)
                    prob_sum = subseq_prob_sum
                    subseq_idx_list.append(subseq_idxs)

                prob_list.append(prob_sum)
            
            # Entropy weighted with or without similarity
            if self.NORM:
                mean_prob = np.mean(prob_list)
                std_prob = np.std(prob_list)
                prob_list = [(prob_list[i] - mean_prob) / std_prob for i in range(len(self.candidate_list))]

            # norm_dist = [1/(1+math.exp(x)) for x in norm_dist]
            score_list = []
            for i in range(len(self.candidate_list)):
                if self.METHOD == 'none':
                    score_list.append(prob_list[i])
                else:
                    score_list.append(prob_list[i] * math.pow(distance[i], self.BETA))
            
            # Locate the subseq_idx with largest score
            rank_idx = np.argsort(np.array(score_list), kind='mergesort').tolist()[::-1]
            for i in rank_idx:
                if i not in visited_candidate_idx:
                    seq_idx = i
                    visited_candidate_idx.append(seq_idx)
                    break
            query_seq = self.candidate_list[seq_idx]

            if self.STRATEGY == 'partial':
                subseq_idxs = subseq_idx_list[seq_idx]
                predict_y = self.crf.predict(query_seq)
                for i in range(len(query_seq[1])):
                    if i not in subseq_idxs:
                        query_seq[1][i] = predict_y[i]
                self.count += len(subseq_idxs)
            else:
                self.count += len(query_seq[1])
            self.cost_list.append(self.count)

            self.update_model([query_seq])
            (in_acc, out_acc, all_acc) =self.crf.evaluate_acc(self.validation_list)
            self.in_acc_list.append(in_acc)
            self.out_acc_list.append(out_acc)
            self.all_acc_list.append(all_acc)
            iterator.close()

        # instead of printing, save model?
        print ('Done!') 

        print("COST_LIST: ", self.cost_list)
        print("IN_ACC_LIST: ", self.in_acc_list)
        print("OUT_ACC_LIST: ", self.out_acc_list)
        print("ALL_ACC_LIST: ", self.all_acc_list)

    def update_model(self, new_srcids):
        self.crf.add_instances(new_srcids)
        self.crf.train()

    # ESSENTIAL
    def predict_proba(self, target_srcids=None, output_format='ttl', *args, **kwargs):
        res = self.predict(target_srcids, output_format)
        return res

    # ESSENTIAL
    def update_prior(self, pred_g, pred_confidences={}):   
        return   
    # ESSENTIAL
    def predict(self, target_srcids=None, output_format='ttl', *args, **kwargs):  
        return self.crf.predict(target_srcids)

    def save_model(self, filename):
        filehandler = open(filename, 'wb')
        pickle.dump(self.crf, filehandler)
    def load_model(self,filename):
        filehandler = open(filename, 'rb')
        return pickle.load(filehandler)

# CRF
class CrfModel(object):
    def __init__(self, data):
        self.label_dict = data.label_dict
        self.word_dict = data.word_dict
        
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        
        self.X_train=[]
        self.Y_train=[]
    
        print ('label dict size: {}'.format(len(self.label_dict)))
        print ('word dict size: {}'.format(len(self.word_dict)))
        
    def reset(self):
        self.X_train=[]
        self.Y_train=[]
    
    def char2feature(self, sent, i):
        # for current character
        features = {'0:word': sent[i]}
        # for previous character
        if i > 0:
            features.update({'-1:word': sent[i-1]})
        # for next character
        if i < len(sent)-1:
            features.update({'+1:word': sent[i+1]})
        return features
    
    def add_instances(self, sequences):
        for seq in sequences:
            x = seq[0]
            y = seq[1]
            self.X_train.append([self.char2feature(x, i) for i in range(len(x))])
            self.Y_train.append(y)
    
    def compute_confidence(self, sequence):
        x = [self.char2feature(sequence[0], i) for i in range(len(sequence[0]))]
        y_pred = self.crf.tagger_.tag(x)
        prob_norm = math.exp(math.log(self.crf.tagger_.probability(y_pred)) / len(x))
        
        label_list = self.crf.tagger_.labels()
        prob_list = []
        for i in range(len(x)):
            marginal_prob = [self.crf.tagger_.marginal(k, i) for k in label_list]
            prob_list.append(max(marginal_prob))
        return (prob_list, sum(prob_list), prob_norm)
    
    def compute_entropy(self, sequence):
        x = [self.char2feature(sequence[0], i) for i in range(len(sequence[0]))]
        label_list = self.crf.tagger_.labels()
        self.crf.tagger_.set(x)
        entropy_seq = []
        for i in range(len(x)):
            marginal_prob = [self.crf.tagger_.marginal(k, i) for k in label_list]
            entropy_seq.append(scipy.stats.entropy(marginal_prob))
        return (entropy_seq, sum(entropy_seq))
    
    def train(self):
        self.crf.fit(self.X_train, self.Y_train) 
        return len(self.X_train)
    
    def predict(self, sequence):
        x = [self.char2feature(sequence[0], i) for i in range(len(sequence[0]))]
        return self.crf.tagger_.tag(x)    
    
    def evaluate_acc(self, sequences):
        # Calculate phrase-level accuracy and out-of-phrase accuracy
        X_test = [[self.char2feature(seq[0], i) for i in range(len(seq[0]))] for seq in sequences]
        Y_test = [seq[1] for seq in sequences]
        Y_pred = self.crf.predict(X_test)
        
        # Consider the accuracy in phrase level.
        in_cnt,  in_crt = 0, 0    # Total/correct number of phrases
        out_cnt, out_crt = 0, 0   # Total/correct number of "o"
        all_cnt, all_crt = 0, 0   # Total/correct number of all words

        for y_test, y_pred in zip(Y_test, Y_pred):
            correct_flag = False
            for j in range(len(y_test)):
                all_cnt += 1
                if y_test[j] == y_pred[j]:
                    all_crt += 1

                # If the character is a beginning-of-phrase.
                if y_test[j][0] == 'B' and y_test[j] != 'B_none':
                    in_cnt += 1
                    if y_test[j] == y_pred[j]:
                        if correct_flag:
                            in_crt += 1
                        correct_flag = True
                    else:
                        if correct_flag:
                            if y_pred[j][2:] != y_pred[j-1][2:]:  # special case
                                in_crt += 1
                        correct_flag = False

                # If the character is an inside-of-phrase.
                elif y_test[j][0] == 'I':
                    if y_test[j] != y_pred[j]:
                        correct_flag = False

                # If the character is an out-of-phrase.
                elif y_test[j] == 'B_none':
                    out_cnt += 1
                    if y_test[j] == y_pred[j]:
                        out_crt += 1
                        if correct_flag:
                            in_crt += 1
                            correct_flag = False
                    else:
                        if correct_flag:
                            if y_pred[j][2:] != y_pred[j-1][2:]:  # special case
                                in_crt += 1
                            correct_flag = False
            # For the case where the phrase is at the end of a string.
            if correct_flag:
                in_crt += 1
        in_acc = 0 if in_cnt == 0 else in_crt/in_cnt
        out_acc = 0 if out_cnt == 0 else out_crt/out_cnt
        all_acc = 0 if all_cnt == 0 else all_crt/all_cnt 
            
        return in_acc, out_acc, all_acc
    
# Vectorize a set of string by n-grams.
def string_vectorize(Xs_list):
    vc = CV(analyzer='char_wb', ngram_range=(3, 4), min_df=1, token_pattern='[a-z]{2,}')
    name = []
    for i in Xs_list:
        s = re.findall('(?i)[a-z]{2,}', "".join(str(x) for x in i))
        name.append(' '.join(s))
    vc.fit(name)
    vec = vc.transform(name).toarray()
    dictionary = vc.get_feature_names()
    return vec, dictionary


