import os
import sys
import importlib.util
import pdb
import time
import random
import re
from copy import deepcopy
import numpy as np
from functools import reduce
from collections import defaultdict
import arrow

import scipy
from scipy.cluster.vq import *
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as hier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from . import Inferencer
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/zodiac')
from ..metadata_interface import *
from ..common import *
from ..rdf_wrapper import *
from jasonhelper import bidict

POINT_POSTFIXES = ['sensor', 'setpoint', 'alarm', 'command', 'meter']

DEBUG = False

def tokenizer(s):
    return re.findall('[a-z]+', s.lower())

def isemptystr(s):
    if s:
        return False
    else:
        return True

def is_nonempty_item_included(l):
    if False in list(map(isemptystr, l)):
        return True
    else:
        return False

class ZodiacInterface(Inferencer):

    def __init__(self,
                 target_building,
                 target_srcids,
                 source_buildings=[],
                 ui=None,
                 config={}):
        super(ZodiacInterface, self).__init__(
            target_building=target_building,
            source_buildings=source_buildings,
            target_srcids=target_srcids,
            ui=None,
            required_label_types=[POINT_TAGSET],
            target_label_type=POINT_TAGSET,
            config=config,
            framework_name='zodiac')

        # init config file for Zodiac
        if 'n_estimators' not in config:
            self.config['n_estimators'] = 400
        if 'random_state' not in config:
            self.config['random_state'] = 0
        if 'sample_num_list' in config:
            sample_num_list = config['sample_num_list']
        else:
            sample_num_list = [0] * (len(source_buildings) + 1) # +1 for target
        if len(self.source_buildings) > len(sample_num_list):
            sample_num_list.append(0)

        # get srcids from other buildings
        source_buildings_srcids = []
        for source_building, sample_num in zip(source_buildings,
                                               sample_num_list):
            objects = LabeledMetadata.objects(building=source_building)
            try:
                source_srcids = random.sample(
                    [obj.srcid for obj in objects], sample_num)
            except:
                pdb.set_trace()
            source_buildings_srcids += source_srcids

        self.total_srcids = deepcopy(target_srcids) + source_buildings_srcids
        self.ground_truths = {} #just for debugging
        self.true_labels = {}
        self.available_srcids = []
        self.training_labels = []
        self.trained_cids = []

        # Init raw data for Zodiac
        names = {}
        descs = {}
        type_strs = {}
        types = {}
        jci_names = {}
        units = {}
        #for raw_point in RawMetadata.objects(building=self.target_building):
        for srcid in self.total_srcids:
            #srcid = raw_point['srcid']
            #if srcid in self.target_srcids:
            raw_point = RawMetadata.objects(srcid=srcid).first()
            metadata = raw_point['metadata']
            if not metadata:
                raise Exception('Metadata for {0} does not exist'
                                .format(srcid))
            if 'BACnetName' in metadata:
                bacnet_name = metadata['BACnetName']
            else:
                bacnet_name = ''
            names[srcid] = bacnet_name
            if 'VendorGivenName' in metadata:
                vendor_given_name = metadata['VendorGivenName']
            else:
                vendor_given_name = ''
            jci_names[srcid] = vendor_given_name
            if 'BACnetDescription' in metadata:
                bacnet_desc = metadata['BACnetDescription']
            else:
                bacnet_desc = ''
            descs[srcid] = bacnet_desc

            if 'BACnetTypeStr' in metadata:
                bacnet_typestr = {metadata['BACnetTypeStr']: 1}
            else:
                bacnet_typestr = {}
            type_strs[srcid] = bacnet_typestr

            if 'BACnetType' in metadata:
                bacnet_type = {str(metadata['BACnetType']): 1}
            else:
                bacnet_type = {}
            types[srcid] = {str(bacnet_type): 1}
            if 'BACnetUnit' in metadata:
                bacnet_unit = {str(metadata['BACnetUnit']): 1}
            else:
                bacnet_unit = {}
            units[srcid] = bacnet_unit
            label_doc = LabeledMetadata.objects(srcid=srcid).first()
            self.ground_truths[srcid] = label_doc.point_tagset
        self.total_bow = self.init_bow(self.total_srcids,
                                       names,
                                       descs,
                                       units,
                                       type_strs,
                                       types,
                                       jci_names)
        target_bow = self.get_sub_bow(self.target_srcids)
        self.cluster_map = self.create_cluster_map(target_bow,
                                                   self.target_srcids)
        #self.cluster_map = self.create_cluster_map(self.total_bow,
        #                                           self.total_srcids)

        if 'seed_srcids' in config:
            seed_srcids = config['seed_srcids']
        else:
            if 'seed_num' in config:
                seed_num = config['seed_num']
            else:
                seed_num = 10
            seed_srcids = self.get_random_learning_srcids(seed_num)

        self.thresholds = [(0.1,0.95), (0.1,0.9) , (0.15,0.9), (0.15,0.85),
                           (0.2,0.85), (0.25,0.85), (0.3,0.85), (0.35,0.85),
                           (0.4,0.85), (0.45,0.85), (0.5,0.85), (0.55,0.85),
                           (0.6,0.85), (0.65,0.85), (0.7,0.85), (0.75,0.85),
                           (0.8,0.85), (0.84999999,0.85) ]
        self.th_ptr= 0
        self.th_min, self.th_max = self.thresholds[self.th_ptr]

        self.init_model()
        self.update_model(seed_srcids)
        self.available_srcids += source_buildings_srcids
        self.training_labels += [LabeledMetadata.objects(srcid=srcid)
                                  .first().point_tagset
                                  for srcid in source_buildings_srcids]
        self.learn_model()

    def init_model(self):
        self.model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            random_state=self.config['random_state'],
            #n_jobs=conf['n_jobs']
        )

    def update_thresholds(self):
        self.th_ptr += 1
        self.th_min, self.th_max = self.thresholds[self.th_ptr]

    def get_random_learning_srcids(self, sample_num):
        srcids = []
        random_cids = random.sample(self.cluster_map.keys(), sample_num)
        for c_id in random_cids:
            srcid = random.choice(self.cluster_map[c_id])
            srcids.append(srcid)
        return srcids

    def vectorize(self, d, srcids, vectorizer):
        data = [d[srcid] for srcid in srcids]
        if is_nonempty_item_included(data):
            vect = vectorizer.fit_transform(data).toarray()
            return vect
        else:
            return None

    def init_bow(self, srcids,
                 names, descs, units, type_strs, types, jci_names):
        count_vectorizer = CountVectorizer(tokenizer=tokenizer)
        dict_vectorizer = DictVectorizer()

        vectors = [
            self.vectorize(names, srcids, deepcopy(count_vectorizer)),
            self.vectorize(descs, srcids, deepcopy(count_vectorizer)),
            self.vectorize(jci_names, srcids, deepcopy(count_vectorizer)),
            self.vectorize(units, srcids, deepcopy(dict_vectorizer)),
            self.vectorize(type_strs, srcids, deepcopy(dict_vectorizer)),
        ]
        bow = np.hstack([vect for vect in vectors
                         if isinstance(vect, np.ndarray)])
        return bow

    def create_cluster_map(self, bow, srcids):
        cluster_map = {}
        z = linkage(bow, metric='cityblock', method='complete')
        dists = list(set(z[:,2]))
        thresh = (dists[1] + dists[2]) /2
        #thresh = (dists[2] + dists[3]) /2
        print("Threshold: ", thresh)
        b = hier.fcluster(z,thresh, criterion='distance')
        assert bow.shape[0] == len(b)
        assert len(b) == len(srcids)
        for cid, srcid in zip(b, srcids):
            cluster_map[cid] = cluster_map.get(cid, []) + [srcid]

        print('# of clusters: {0}'.format(len(b)))
        print('sizes of clustsers:', sorted(map(len, cluster_map.values())))

        return cluster_map


    def find_cluster_id(self, srcid):
        for cid, srcids in self.cluster_map.items():
            if srcid in srcids:
                return cid
        raise Exception('Srcid not found in the cluster map: {0}'
                        .format(srcid))


    def get_sub_bow(self, srcids):
        return self.total_bow[
            [self.total_srcids.index(srcid) for srcid in srcids]
        ]


    def add_cluster_label(self, cid, label):
        if cid in self.trained_cids:
            print('WARNING: Cluster already learned: {0}'.format(cid))
            return None
        self.trained_cids.append(cid)
        cluster_srcids = self.cluster_map[cid]
        self.available_srcids += cluster_srcids
        self.training_labels += [label] * len(cluster_srcids)
        if DEBUG:
            for srcid in cluster_srcids:
                true_label = self.ground_truths[srcid]
                if true_label != label:
                    print('At {0}, pred({1}) != true({2})'
                          .format(srcid, label, true_label))
                    cluster_all_labels = [self.ground_truths[srcid]
                                          for srcid in cluster_srcids]
                    print('There are {0} labels here'
                          .format(len(set(cluster_all_labels))))

    def apply_prior_augment_samples(self):
        prior_preds = {}
        if self.prior_g:
            for triple, confidence in self.prior_confidences.items():
                if confidence > self.th_max: # If the prediction is confident
                    srcid = triple[0].split('#')[-1]
                    tagset = triple[2].split('#')[-1]
                    if srcid in self.target_srcids:
                        prior_preds[srcid] = tagset
        return prior_preds


    def update_model(self, new_srcids):
        super(ZodiacInterface, self).update_model(new_srcids)

        # Add new srcids into the training set.
        for srcid in new_srcids:
            labeled = LabeledMetadata.objects(srcid=srcid)
            if not labeled:
                raise Exception('Labels do not exist for {0}'.format(srcid))
            labeled = labeled[0]
            point_tagset = labeled.point_tagset
            if not point_tagset:
                raise Exception('Point Tagset not found at {0}: {1}'
                                .format(srcid, labeled.tagsets))
            self.true_labels[srcid] = point_tagset

        for srcid in new_srcids:
            cid = self.find_cluster_id(srcid)
            cluster_label = self.true_labels[srcid]
            self.add_cluster_label(cid, cluster_label)
        self.learn_model()
        self.select_informative_samples(1)

        prior_preds = self.apply_prior_augment_samples()
        for srcid, point_tagset in prior_preds.items():
            if srcid not in self.available_srcids:
                cid = self.find_cluster_id(srcid)
                cluster_label = point_tagset
                self.add_cluster_label(cid, cluster_label)
        if prior_preds:
            self.learn_model()
            self.select_informative_samples(1)

    def select_srcid_per_cluster(self, srcids):
        cids = []
        for srcid in srcids:
            srcid_handled = False
            for cid, cluster in self.cluster_map.items():
                if srcid in cluster:
                    if cid not in self.trained_cids:
                        cids.append(cid)
                    srcid_handled = True
                    break
            assert srcid_handled, "{0}'s cluster is not found".format(srcid)
        new_srcids = []
        cids = list(set(cids))
        cluster_sizes = [len(self.cluster_map[cid]) for cid in cids]
        for cid in cids:
            new_srcids.append(random.choice(self.cluster_map[cid]))
        new_srcids = [row[1] for row in sorted(zip(cluster_sizes, new_srcids),
                                               reverse=True)]
        return new_srcids

    def apply_prior_quiver(self, pred, target_srcids):
        if not self.prior_g:
            return []

        # If points in a vav are identified same,
        # remove it from identified list.
        vavs = get_vavs(self.prior_g)
        cand_srcids = []
        for vav in vavs:
            points = get_vav_points(self.prior_g, vav)
            point_types = defaultdict(list)
            for point in points:
                srcid = point.split('#')[-1]
                if srcid in target_srcids:
                    point_idx = target_srcids.index(srcid)
                    pred_type = pred[point_idx]
                    point_types[pred_type].append(point)
            for point_type, points in point_types.items():
                if len(points) > 2:
                    cand_srcids += [parse_srcid(point) for point in points]
        new_srcids = self.select_srcid_per_cluster(cand_srcids)
        return new_srcids

    def select_informative_samples(self, sample_num=1):
        new_srcids = []
        tot_srcids = reduce(adder, self.cluster_map.values())
        base_sample_bow = self.get_sub_bow(tot_srcids)
        base_confidence = self.model.predict_proba(base_sample_bow)
        base_pred_labels = self.model.predict(base_sample_bow)
        new_srcids = self.apply_prior_quiver(base_pred_labels, tot_srcids)
        new_srcids = new_srcids[0:sample_num]

        #th_update_flag and \
        test_flag = 0
        looping_flag = False
        while\
                len(self.available_srcids) != len(self.total_srcids) and \
                len(new_srcids) < sample_num:
            self.learn_model()
            th_update_flag = True
            prev_available_srcids = deepcopy(self.available_srcids)
            print('curr availble srcids: {0}'
                  .format(len(prev_available_srcids)))
            for cid, cluster_srcids in self.cluster_map.items():
                if cid in self.trained_cids:
                    continue
                sample_bow = self.get_sub_bow(cluster_srcids)
                confidence = self.model.predict_proba(sample_bow)
                pred_labels = self.model.predict(sample_bow)
                max_confidence = 0
                max_confidence = max(map(max, confidence))

                if max_confidence >= self.th_min and \
                        max_confidence < self.th_max: # Gray zone
                    pass
                elif max_confidence >= self.th_max:
                    if looping_flag:
                        pdb.set_trace()
                    th_update_flag = False
                    test_flag = cluster_srcids
                    self.trained_cids.append(cid)
                    if cluster_srcids[0] in prev_available_srcids:
                        pdb.set_trace()
                    self.available_srcids += cluster_srcids
                    self.training_labels += pred_labels.tolist()
                    # Check true label for debugging
                    if DEBUG:
                        for srcid, pred_label in zip(cluster_srcids,
                                                     pred_labels):
                            true_label = self.ground_truths[srcid]
                            if true_label != pred_label:
                                print('At {0}, pred({1}) != true({2})'
                                      .format(srcid, pred_label, true_label))
                                pdb.set_trace()
                    break
                elif max_confidence < self.th_min:
                    if looping_flag:
                        pdb.set_trace()
                    test_flag = 2
                    new_srcids.append(random.choice(cluster_srcids))
                    th_update_flag = False
                    if len(new_srcids) ==  sample_num:
                        break

            if th_update_flag:
                print('th updated')
                #if not (len(self.available_srcids) > len(temp_available_srcids)\
                #        or len(new_srcids) > 0):
                #    pdb.set_trace()
                self.update_thresholds()
            else:
                print('th not updated because ')
                if len(new_srcids) > 0:
                    print('new srcids are found: {0}'.format(len(new_srcids)))
                elif len(self.available_srcids) > len(prev_available_srcids):
                    print('increased srcids: {0}'.format(
                        len(self.available_srcids) - len(prev_available_srcids)))
                else:
                    print('test flag: {0}'.format(test_flag))
                    looping_flag = True

            print('Current threshold pointer: {0}/{1}'
                  .format(self.th_ptr, len(self.thresholds)))
        return new_srcids

    def get_num_sensors_in_gray(self):
        # TODO: This line should consider source building srcids"
        return len(self.target_srcids) - len(self.available_srcids)

    def custom_accuracy(self):
        sample_bow = self.get_sub_bow(self.target_srcids)
        pred_labels = self.model.predict(sample_bow)
        acc = sum([pred_label == self.ground_truths[srcid]
                   for srcid, pred_label
                   in zip(self.target_srcids, pred_labels)]) / \
            len(self.target_srcids)
        return acc

    def learn_auto(self):
        gray_num = 1000
        cnt = 0
        while gray_num > 0:
            print('--------------------------')
            print('{0}th iteration'.format(cnt))
            self.learn_model()
            new_srcids = self.select_informative_samples(1)
            self.update_model(new_srcids)
            gray_num = self.get_num_sensors_in_gray()
            self.evaluate(self.target_srcids)
            acc = self.custom_accuracy()
            print('curr new srcids: {0}'.format(len(new_srcids)))
            if new_srcids:
                print('new cluster\'s size: {0}'
                      .format(len(self.cluster_map[self.find_cluster_id(
                          new_srcids[0])])))
            print('training srcids: {0}'.format(len(self.training_srcids)))
            print('gray: {0}/{1}'.format(gray_num, len(self.target_srcids)))
            print('f1: {0}'.format(self.history[-1]['metrics']['f1']))
            print('macrof1: {0}'.format(self.history[-1]['metrics']['macrof1']))
            cnt += 1
        self.learn_model()

    def learn_model(self):
        self.training_bow = self.get_sub_bow(self.available_srcids)
        self.model.fit(self.training_bow, self.training_labels)

    def predict(self, target_srcids=None):
        t0 = arrow.get()
        if not target_srcids:
            target_srcids = self.target_srcids
        super(ZodiacInterface, self).predict(target_srcids)

        self.learn_model()

        pred_g = None
        for i in range(0, 10):
            try:
                pred_g = init_graph(empty=False)
            except:
                print('WARNING: init_graph temporarily failed.')
            if pred_g:
                break
            time.sleep(3)
        assert pred_g, 'pred_g is not initialized somehow'
        sample_bow = self.get_sub_bow(target_srcids)

        pred_points = self.model.predict(sample_bow)
        confidences = self.model.predict_proba(sample_bow)
        for srcid, pred_point, prob in zip(target_srcids,
                                           pred_points,
                                           confidences):
            prob = max(prob)
            try:
                self._add_pred_point_result(pred_g, srcid, pred_point, prob)
            except Exception as e:
                print('ERROR: {0}'.format(e))
                pdb.set_trace()
        self.pred_g = pred_g
        t1 = arrow.get()
        print('REALLY it takes this: {0}'.format(t1 - t0))
        return pred_g
