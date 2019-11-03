import json
import os
import argparse
import random
from functools import reduce, partial
import logging
import re
from collections import defaultdict, OrderedDict
import pdb
import sys
import requests
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as hier

from .eval_func import *

POINT_POSTFIXES = ['sensor', 'setpoint', 'command', 'alarm', 'status', 'meter']


def elem2list(elem):
    if isinstance(elem, str):
        return elem.split('_')
    else:
        return []


def csv2json(df, key_idx, value_idx):
    keys = df[key_idx].tolist()
    values = df[value_idx].tolist()
    return {k: elem2list(v) for k, v in zip(keys, values)}


def sub_dict_by_key_set(d, ks):
    return dict((k, v) for k, v in d.items() if k in ks)


def leave_one_word(s, w):
    if w in s:
        s = s.replace(w, '')
        s = w + '-' + s
    return s


def find_keys(tv, d, crit=lambda x, y: x == y):
    keys = list()
    for k, v in d.items():
        if crit(tv, v):
            keys.append(k)
    return keys


def check_in(x, y):
    return x in y


def joiner(s):
    return ''.join(s)


def get_word_clusters(sentence_dict):
    srcids = list(sentence_dict.keys())
    sentences = []
    for srcid in srcids:
        sentence = []
        for metadata_type, sent in sentence_dict[srcid].items():
            sentence.append(''.join(sent))
        sentence = '\n'.join(sentence)
        sentence = ' '.join(re.findall('[a-z]+', sentence))
        sentences.append(sentence)
    vect = TfidfVectorizer()
    bow = vect.fit_transform(sentences).toarray()
    z = linkage(bow, metric='cityblock', method='complete')
    dists = list(set(z[:, 2]))
    thresh = (dists[2] + dists[3]) / 2
    #thresh = (dists[1] + dists[2]) /2
    print("Threshold: ", thresh)
    b = hier.fcluster(z, thresh, criterion='distance')
    cluster_dict = defaultdict(list)

    for srcid, cluster_id in zip(srcids, b):
        cluster_dict[cluster_id].append(srcid)
    return dict(cluster_dict)


def select_random_samples(building,
                          srcids,
                          n,
                          use_cluster_flag,
                          sentence_dict=None,
                          token_type='justseparate',
                          reverse=True,
                          cluster_dict=None,
                          shuffle_flag=True,
                          unique_clusters_flag=False,
                          ):
    assert sentence_dict or cluster_dict
    if not cluster_dict:
        cluster_dict = get_word_clusters(sentence_dict)

    # Learning Sample Selection
    sample_srcids = set()
    length_counter = lambda x: len(x[1])
    if use_cluster_flag:
        sorted_cluster_dict = OrderedDict(
            sorted(cluster_dict.items(), key=length_counter, reverse=reverse))
        #n = len(sorted_cluster_dict) #TODO: Remove if not working well
        while len(sample_srcids) < n:
            cluster_dict_items = list(sorted_cluster_dict.items())
            if shuffle_flag:
                random.shuffle(cluster_dict_items)
            for cluster_num, srcid_list in cluster_dict_items:
                valid_srcid_list = set(srcid_list)\
                    .intersection(set(srcids))\
                    .difference(set(sample_srcids))
                if len(valid_srcid_list) > 0:
                    sample_srcids.add(random.choice(list(valid_srcid_list)))
                if len(sample_srcids) >= n:
                    break
            if unique_clusters_flag:
                break
    else:
        sample_srcids = random.sample(srcids, n)
    return list(sample_srcids)

def splitter(s):
    return s.split('_')

def alpha_tokenizer(s): 
    return re.findall('[a-zA-Z]+', s)

def adder(x, y):
    return x+y

def bilou_tagset_phraser(sentence, token_labels, keep_alltokens=False):
    phrase_labels = list()
    curr_phrase = ''
    for i, (c, label) in enumerate(zip(sentence, token_labels)):
        if label[2:] in ['rightidentifier', 'leftidentifier'] \
                and not keep_alltokens:
            continue
        try:
            tag = label[0]
        except:
            pdb.set_trace()
        if tag=='B':
            if curr_phrase:
            # Below is redundant if the other tags handles correctly.       
                phrase_labels.append(curr_phrase)
            curr_phrase = label[2:]
        elif tag == 'I':
            if curr_phrase != label[2:] and curr_phrase:
                phrase_labels.append(curr_phrase)
                curr_phrase = label[2:]
        elif tag == 'L':
            if curr_phrase != label[2:] and curr_phrase:
                # Add if the previous label is different                    
                phrase_labels.append(curr_phrase)
            # Add current label                                             
            phrase_labels.append(label[2:])
            curr_phrase = ''
        elif tag == 'O':
            # Do nothing other than pushing the previous label
            if not keep_alltokens:
                if curr_phrase:
                    phrase_labels.append(curr_phrase)
                curr_phrase = ''
            else:
                if curr_phrase == 'O':
                    pass
                else:
                    if curr_phrase:
                        phrase_labels.append(curr_phrase)
                    curr_phrase = 'O'

        elif tag == 'U':
            if curr_phrase:
                phrase_labels.append(curr_phrase)
            phrase_labels.append(label[2:])
        else:
            print('Tag is incorrect in: {0}.'.format(label))
            pdb.set_trace()
        if len(phrase_labels)>0:
            if phrase_labels[-1] == '':
                pdb.set_trace()
    if curr_phrase != '':
        phrase_labels.append(curr_phrase)
    phrase_labels = [leave_one_word(\
                         leave_one_word(phrase_label, 'leftidentifier'),\
                            'rightidentifier')\
                        for phrase_label in phrase_labels]
    phrase_labels = list(reduce(adder, map(splitter, phrase_labels), []))
    return phrase_labels

def make_phrase_dict(sentence_dict, token_label_dict, keep_alltokens=False):
    #phrase_dict = OrderedDict()
    phrase_dict = dict()
    for srcid, token_labels_dict in token_label_dict.items():
        phrases = []
        for metadata_type, token_labels in token_labels_dict.items():
            sentence = sentence_dict[srcid][metadata_type]
            phrases += bilou_tagset_phraser(
                sentence, token_labels, keep_alltokens)
        remove_indices = list()
        for i, phrase in enumerate(phrases):
            #TODO: Below is heuristic. Is it allowable?
            #if phrase.split('-')[0] in ['building', 'networkadapter',\
            #                            'leftidentifier', 'rightidentifier']:
            if phrase.split('-')[0] in ['leftidentifier', 'rightidentifier']\
                    and not keep_alltokens:
                pdb.set_trace()
                remove_indices.append(i)
                pass
        phrases = [phrase for i, phrase in enumerate(phrases)\
                   if i not in remove_indices]
        #phrase_dict[srcid] = phrases + phrases # TODO: Why did I put this before?
        phrase_dict[srcid] = phrases
    return phrase_dict

def hier_clustering(d, threshold=3):
    srcids = d.keys()
    tokenizer = lambda x: x.split()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    assert isinstance(d, dict)
    assert isinstance(list(d.values())[0], list)
    assert isinstance(list(d.values())[0][0], str)
    doc = [' '.join(d[srcid]) for srcid in srcids]
    vect = vectorizer.fit_transform(doc)
    #TODO: Make vect aligned to the required format
    z = linkage(vect.toarray(), metric='cityblock', method='complete')
    dists = list(set(z[:,2]))
#    threshold = 3
    #threshold = (dists[2] + dists[3]) / 2
    b = hier.fcluster(z, threshold, criterion='distance')
    cluster_dict = defaultdict(list)
    for srcid, cluster_id in zip(srcids, b):
        cluster_dict[str(cluster_id)].append(srcid)
    value_lengther = lambda x: len(x[1])
    return OrderedDict(\
               sorted(cluster_dict.items(), key=value_lengther, reverse=True))

def set_logger(logfile=None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # Console Handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    # File Handler
    if logfile:
        fh = logging.FileHandler(logfile, mode='w+')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger


def parallel_func(orig_func, return_idx, return_dict, *args):
    return_dict[return_idx] = orig_func(*args)


def iteration_wrapper(iter_num, func, prev_data=None, *params):
    step_datas = list()
    if not prev_data:
        prev_data = {'iter_num':0,
                     'learning_srcids': [],
                     'model_uuid': None}
    for i in range(0, iter_num):
        print('{0} th stage started'.format(prev_data['iter_num']))
        step_data = func(prev_data, *params)
        print('{0} th stage finished'.format(prev_data['iter_num']))
        step_datas.append(step_data)
        prev_data = step_data
        prev_data['iter_num'] += 1
    return step_datas


def replace_num_or_special(word):
    if re.match('\d+', word):
        return 'NUMBER'
    elif re.match('[a-zA-Z]+', word):
        return word
    else:
        return 'SPECIAL'


def adder(x, y):
    return x + y


def get_cluster_dict(building):
    cluster_filename = 'model/%s_word_clustering_justseparate.json' % (building.id)
    with open(cluster_filename, 'r') as fp:
        cluster_dict = json.load(fp)
    return cluster_dict


def get_label_dict(building):
    raise Exception('This should be replaced')
    filename = 'metadata/%s_label_dict_justseparate.json' % (building.id)
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


def merge_sentences(self, sentences):
    return {
        'VendorGivenName': '@\t@'.join(['@'.join(sentences[column])
                                        for column in column_names
                                        if column in sentences]).split('@')
    }


def merge_labels(labels):
    return {
        srcid: {
            'VendorGivenName': '@O@'.join(['@'.join(one_labels[column])
                                           for column in column_names
                                           if column in one_labels]).split('@')
        }
        for srcid, one_labels in labels.items()
    }


def find_points(tagsets):
    points = []
    for tagset in tagsets:
        postfix = tagset.split('_')[-1]
        if postfix in POINT_POSTFIXES:
            points.append(tagset)
    if not points:
        points = ['none']
    return set(points)
