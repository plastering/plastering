from uuid import uuid4
from functools import reduce
import pdb
import os
from operator import itemgetter
from itertools import chain
from copy import deepcopy

import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, chi2, SelectPercentile, SelectKBest
from sklearn.pipeline import Pipeline
from scipy.sparse import vstack, csr_matrix, hstack, issparse, coo_matrix, \
    lil_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import LabelPowerset, ClassifierChain, \
    BinaryRelevance
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import entropy as get_entropy


from .time_series_to_ir import TimeSeriesToIR
from .base_scrabble import BaseScrabble
from .common import *
from .hcc import StructuredClassifierChain
from .brick_parser2 import get_subclasses, get_subclasses_dict, get_tagset_tree
#from .brick_parser import tagsetTree as tagset_tree
from .dann import DANN

from keras.layers import Input, Dense, Dropout
from keras.models import Sequential
from keras.constraints import max_norm
from keras import regularizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def gen_uuid():
    return str(uuid4())



def tree_flatter(tree, init_flag=True):
    branches_list = list(tree.values())
    d_list = list(tree.keys())
    for branches in branches_list:
        for branch in branches:
            added_d_list = tree_flatter(branch)
            d_list = [d for d in d_list if d not in added_d_list]\
                    + added_d_list
    return d_list

def extend_tree(tree, k, d):
    for curr_head, branches in tree.items():
        if k==curr_head:
            branches.append(d)
        for branch in branches:
            extend_tree(branch, k, d)

def calc_leaves_depth(tree, d=dict(), depth=0):
    curr_depth = depth + 1
    for tagset, branches in tree.items():
        if d.get(tagset):
            d[tagset] = max(d[tagset], curr_depth)
        else:
            d[tagset] = curr_depth
        for branch in branches:
            new_d = calc_leaves_depth(branch, d, curr_depth)
            for k, v in new_d.items():
                if d.get(k):
                    d[k] = max(d[k], v)
                else:
                    d[k] = v
    return d

def augment_tagset_tree(tagsets, subclass_dict, tagset_tree):
    for tagset in set(tagsets):
        if '-' in tagset:
            classname = tagset.split('-')[0]
            extend_tree(tagset_tree, classname, {tagset:[]})
            try:
                subclass_dict[classname].append(tagset)
            except:
                pdb.set_trace()
            subclass_dict[tagset] = []
        else:
            if tagset not in subclass_dict.keys():
                classname = tagset.split('_')[-1]
                try:
                    subclass_dict[classname].append(tagset)
                except:
                    pdb.set_trace()
                subclass_dict[tagset] = []
                extend_tree(tagset_tree, classname, {tagset:[]})


class Ir2Tagsets(BaseScrabble):
    """docstring for Ir2Tagsets"""

    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 building_tagsets_dict,
                 source_buildings=[],
                 source_sample_num_list=[],
                 learning_srcids=[],
                 known_tags_dict={},
                 pgid=None,
                 config={}):
        super(Ir2Tagsets, self).__init__(
            target_building,
            target_srcids,
            building_label_dict,
            building_sentence_dict,
            building_tagsets_dict,
            source_buildings,
            source_sample_num_list,
            learning_srcids,
            pgid,
            config)
        self.ts2ir = None
        self.ts_feature_filename = 'temp/features.pkl'

        self.known_tags_dict = known_tags_dict
        if 'use_cluster_flag' in config:
            self.use_cluster_flag = config['use_cluster_flag']
        else:
            self.use_cluster_flag = True
        if 'eda_flag' in config:
            self.eda_flag = config['eda_flag'],
        else:
            self.eda_flag = False
        if 'use_brick_flag' in config:
            self.use_brick_flag = config['use_brick_flag']
        else:
            self.use_brick_flag = True
        if 'n_jobs' in config:
            self.n_jobs = config['n_jobs']
        else:
            #self.n_jobs = 1
            self.n_jobs = 6
        if 'negative_flag' in config:
            self.negative_flag = config['negative_flag']
        else:
            self.negative_flag = True
        if 'emptydoc_flag' in config:
            self.emptydoc_flag = config['emptydoc_flag']
        else:
            self.emptydoc_flag = True
        if 'tagset_classifier_type' in config:
            self.tagset_classifier_type = config['tagset_classifier_type']
        else:
            self.tagset_classifier_type = 'MLP'
        if 'n_estimators' in config:
            self.n_estimators = config['n_estimators']
        else:
            self.n_estimators = 10 # TODO: Find the proper value
        if 'vectorizer_type' in config:
            self.vectorizer_type = config['vectorizer_type']
        else:
            #self.vectorizer_type = 'count'
            self.vectorizer_type = 'tfidf'
        if 'entqs' in config:
            self.query_strategy = config['entqs']
        else:
            self.query_strategy = 'entropy'
        if 'use_known_tags' in config:
            self.use_known_tags = config['use_known_tags']
        else:
            self.use_known_tags = False
        if 'expand_tagsets_by_hierarchy_flag' in config:
            self.expand_tagsets_by_hierarchy_flag = config['expand_tagsets_by_hierarchy_flag']
        else:
            self.expand_tagsets_by_hierarchy_flag = True

        self.epochs = config.get('ir2tagsets.epochs', 400)
        self.nb_empty_docs = 50

        self._init_brick()
        self._init_data(learning_srcids)

    def _init_brick(self):
        self.brick_srcids = []
        version = '1.0.3' #TODO: Parameterize it from the module
        self.tagset_list = get_subclasses(version, 'bf:TagSet')
        self.point_tagsets = get_subclasses(version, 'brick:Point')
        self.tagset_list.append('networkadapter')

        self.subclass_dict = get_subclasses_dict(version, 'bf:TagSet')
        self.subclass_dict['networkadapter'] = list()
        self.subclass_dict['unknown'] = list()
        self.subclass_dict['none'] = list()
        #self.tagset_tree = deepcopy(tagset_tree)
        self.tagset_tree = get_tagset_tree(version)

    def get_srcid_domain(self, srcid):
        # try get building name
        splitted = srcid.split(';')
        if len(splitted) == 1:
            orig_srcid = srcid
        else:
            orig_srcid, srcid_postfix = splitted
        if orig_srcid == 'brick':
            domain = 'brick'
        else:
            domain = None
            for building_id, sentence_dict in self.building_sentence_dict.items():
                if orig_srcid in sentence_dict:
                    domain = str(building_id)
        assert domain
        return domain

    def expand_tagsets_by_hierarchy(self):
        for srcid, tagsets in self.tagsets_dict.items():
            expanded = set(tagsets)
            for tagset in tagsets:
                for superclass, subclasses in self.subclass_dict.items():
                    if tagset in subclasses:
                        expanded.add(superclass)
            self.tagsets_dict[srcid] = list(expanded)


    def _init_data(self, learning_srcids=[]):
        self.sentence_dict = {}
        self.label_dict = {}
        self.tagsets_dict = {}
        self.phrase_dict = {}
        self.point_dict = {}
        self.building_cluster_dict = {}

        for building, source_sample_num in zip(self.source_buildings,
                                               self.source_sample_num_list):
            self.sentence_dict.update(self.building_sentence_dict[building.id])
            one_label_dict = self.building_label_dict[building.id]
            self.label_dict.update(one_label_dict)

            if learning_srcids:
                self.learning_srcids = learning_srcids
            else:
                sample_srcid_list = select_random_samples(
                    building = building.id,
                    srcids = one_label_dict.keys(),
                    n = source_sample_num,
                    use_cluster_flag = self.use_cluster_flag,
                    sentence_dict = self.building_sentence_dict[building.id],
                    shuffle_flag = False
                )
                self.learning_srcids += sample_srcid_list
            one_tagsets_dict = self.building_tagsets_dict[building.id]
            self.tagsets_dict.update(one_tagsets_dict)
            for srcid, tagsets in one_tagsets_dict.items():
                point_tagset = 'none'
                for tagset in tagsets:
                    if tagset in self.point_tagsets:
                        point_tagset = tagset
                        break
                self.point_dict[srcid] = point_tagset
            if building not in self.building_cluster_dict:
                self.building_cluster_dict[building.id] = get_word_clusters(
                    self.building_sentence_dict[building.id])

        self.phrase_dict = make_phrase_dict(self.sentence_dict, 
                                            self.label_dict)
        if self.expand_tagsets_by_hierarchy_flag:
            self.expand_tagsets_by_hierarchy()


    def _extend_tagset_list(self, new_tagsets):
        self.tagset_list += new_tagsets
        self.tagset_list = list(set(self.tagset_list))

    def update_model(self, srcids):
        self.learning_srcids += list(srcids) * 2
        self.target_srcids = [srcid for srcid in self.target_srcids
                              if srcid not in self.learning_srcids]
        # invalid_num = sum([srcid not in self.tagsets_dict for srcid in
        #                    self.learning_srcids + self.target_srcids])  # for debug
        self._extend_tagset_list(reduce(adder, [self.tagsets_dict[srcid] for srcid
                                                in self.learning_srcids]))
        #augment_tagset_tree(self.tagset_list, self.subclass_dict, self.tagset_tree)
        self._build_tagset_classifier(self.learning_srcids,
                                      self.target_srcids,
                                      validation_srcids=[])

    def _determine_used_phrases(self, phrases, tagsets):
        phrases_usages = list()
        pred_tags = reduce(adder, [tagset.split('_') for tagset in tagsets], [])
        used_cnt = 0.0
        unused_cnt = 0.0
        for phrase in phrases:
            phrase_tags = phrase.split('_')
            for tag in phrase_tags:
                if tag in ['leftidentifier', 'rightidentifier']:
                    continue
                if tag in pred_tags:
                    used_cnt += 1 / len(phrase_tags)
                else:
                    unused_cnt += 1 / len(phrase_tags)
        if used_cnt == 0:
            score = 0
        else:
            score = used_cnt / (used_cnt + unused_cnt) 
        return score

    def ir2tagset_al_query_samples_phrase_util(self,
                                               test_srcids,
                                               building,
                                               pred_tagsets_dict,
                                               inc_num,
                                               phrase_dict
                                               ):
        phrase_usage_dict = {}
        for srcid in test_srcids:
            pred_tagsets = pred_tagsets_dict[srcid]
            phrase_usage_dict[srcid] = self._determine_used_phrases(phrase_dict[srcid],
                                                                    pred_tagsets)

        phrase_usages = list(phrase_usage_dict.values())
        mean_usage_rate = np.mean(phrase_usages)
        std_usage_rate = np.std(phrase_usages)
        # Select underexploited sentences.
        threshold = mean_usage_rate - std_usage_rate
        todo_sentence_dict = dict(
            (srcid, alpha_tokenizer(''.join(self.sentence_dict[srcid])))
            for srcid, usage_rate
            in phrase_usage_dict.items()
            if usage_rate < threshold and srcid in test_srcids)
        cluster_dict = self.building_cluster_dict[building.id]
        todo_srcids = select_random_samples(
            building = building,
            srcids = list(todo_sentence_dict.keys()),
            n = min(inc_num, len(todo_sentence_dict)),
            use_cluster_flag = True,
            cluster_dict = cluster_dict,
            shuffle_flag = False,
        )
        #if the numbers are not enough randomly select more:
        if len(todo_srcids) < inc_num:
            more_num = inc_num - len(todo_srcids)
            todo_sentence_dict = dict(
                (srcid, alpha_tokenizer(''.join(self.sentence_dict[srcid])))
                for srcid, usage_rate
                in phrase_usage_dict.items()
                if srcid in test_srcids)
            cluster_dict = self.building_cluster_dict[building.id]
            todo_srcids += select_random_samples(
                building = building,
                srcids = list(todo_sentence_dict.keys()),
                n = min(more_num, len(todo_sentence_dict)),
                use_cluster_flag = True,
                cluster_dict = cluster_dict,
                shuffle_flag = True
            )
        return todo_srcids

    def ir2tagset_al_query_entropy(self,
                                   target_prob_mat,
                                   #target_prob,
                                   target_srcids,
                                   learning_srcids,
                                   target_building,
                                   inc_num
                                   ):
        assert len(target_srcids) == target_prob_mat.shape[0]
        entropies = get_entropy(target_prob_mat.T)
        sorted_entropies = sorted([(srcid, ent) for srcid, ent
                                   in zip(target_srcids, entropies)],
                                  key=itemgetter(1))
        cluster_dict = self.building_cluster_dict[target_building.id]
        added_cids = []
        todo_srcids = []
        new_srcid_cnt = 0
        for srcid, ent in sorted_entropies:
            if srcid in learning_srcids:
                continue
            the_cid = None
            for cid, cluster in cluster_dict.items():
                if srcid in cluster:
                    the_cid = cid
                    break
            if the_cid in added_cids:
                continue
            added_cids.append(the_cid)
            todo_srcids.append(srcid)
            new_srcid_cnt += 1
            if new_srcid_cnt == inc_num:
                break
        return todo_srcids

    def select_informative_samples(self, sample_num, phrase_dict=None):
        if not phrase_dict:
            phrase_dict = deepcopy(self.phrase_dict)
        if self.query_strategy == 'phrase_util':
            pred = self.predict(self.target_srcids, phrase_dict=phrase_dict)
            new_srcids = self.ir2tagset_al_query_samples_phrase_util(
                self.target_srcids,
                self.target_building,
                pred,
                sample_num,
                phrase_dict
            )
        elif self.query_strategy == 'entropy':
            _, _, prob_mat = self._predict_and_proba(self.target_srcids, True)
            #proba = self.predict_proba(self.target_srcids)
            new_srcids = self.ir2tagset_al_query_entropy(
                             prob_mat,
                             self.target_srcids,
                             self.learning_srcids,
                             self.target_building,
                             sample_num)
        else:
            raise ValueError('Query Strategy Wrong: {0}'.format(query_strategy))
        return new_srcids

    # ESSENTIAL
    def learn_auto(self, iter_num=1):
        """Learn from the scratch to the end.
        """
        pass

    def _augment_phrases_with_ts(self, phrase_dict, srcids, ts2ir):
        with open(self.ts_feature_filename, 'rb') as fp:
            ts_features = pickle.load(fp, encoding='bytes')
        ts_tags_pred = ts2ir.predict(ts_features, srcids)

        tag_binarizer = ts2ir.get_binarizer()
        pred_tags_list = tag_binarizer.inverse_transform(ts_tags_pred)

        for srcid, pred_tags in zip(srcids, pred_tags_list):
            phrase_dict[srcid] += list(pred_tags)
        return phrase_dict

    def _predict_and_proba(self, target_srcids, full_prob=False, phrase_dict=None):
        if not target_srcids:
            return {}, {}
        if not phrase_dict:
            phrase_dict = deepcopy(self.phrase_dict)

        #phrase_dict = {srcid: self.phrase_dict[srcid] for srcid in target_srcids}
        if self.use_known_tags:
            doc = [' '.join(phrase_dict[srcid] + self.known_tags_dict[srcid])
                   for srcid in target_srcids]
        else:
            doc = [' '.join(phrase_dict[srcid]) for srcid in target_srcids]
        vect_doc = self.tagset_vectorizer.transform(doc) # should this be fit_transform?

        certainty_dict = dict()
        tagsets_dict = dict()
        if self.tagset_classifier_type in ['MLP', 'DANN']:
            pred_mat = self.tagset_classifier.predict(vect_doc)
            prob_mat = deepcopy(pred_mat)
            pred_mat[pred_mat >= 0.5] = 1
            pred_mat[pred_mat < 0.5] = 0
        else:
            pred_mat = self.tagset_classifier.predict(vect_doc)
            prob_mat = self.tagset_classifier.predict_proba(vect_doc)
        if not isinstance(pred_mat, np.ndarray):
            try:
                pred_mat = pred_mat.toarray()
            except:
                pred_mat = np.asarray(pred_mat)
        pred_tagsets_dict = dict()
        pred_certainty_dict = dict()
        pred_point_dict = dict()
        for i, (srcid, pred, prob) in enumerate(zip(target_srcids, pred_mat, prob_mat)):
            pred_tagsets = self.tagset_binarizer.inverse_transform(np.asarray([pred]))[0]
            #pred_tagsets_dict[srcid] = self.tagset_binarizer.inverse_transform(\
            #                                np.asarray([pred]))[0]
            if self.expand_tagsets_by_hierarchy_flag:
                filtered = deepcopy(list(pred_tagsets))
                for curr_tagset in pred_tagsets:
                    for other_tagset in filtered:
                        if other_tagset in self.subclass_dict[curr_tagset]:
                            filtered.remove(curr_tagset)
                            break
                pred_tagsets = tuple(filtered)
                max_prob = max(prob) #TODO: implement this for filtered ones
            else:
                max_prob = max(prob)
            pred_tagsets_dict[srcid] = pred_tagsets
            pred_certainty_dict[srcid] = max_prob
            #pred_certainty_dict[srcid] = 0
        pred_certainty_dict = OrderedDict(sorted(pred_certainty_dict.items(), \
                                                 key=itemgetter(1), reverse=True))
        logging.info('Finished prediction')
        if full_prob:
            return pred_tagsets_dict, pred_certainty_dict, prob_mat
        else:
            return pred_tagsets_dict, pred_certainty_dict

    def predict(self, target_srcids=None, phrase_dict=None):
        if not target_srcids:
            target_srcids =self.target_srcids
        pred, _ = self._predict_and_proba(target_srcids, phrase_dict=phrase_dict)
        return pred

    def predict_proba(self, target_srcids=None):
        if not target_srcids:
            target_srcids =self.target_srcids
        _, proba =self._predict_and_proba(target_srcids)
        return proba

    def _build_point_classifier(self):
        # TODO: Implement this later if needed
        #       Currently, just collected garbages.
        self.point_classifier = RandomForestClassifier(
                               n_estimators=self.n_estimators,
                               n_jobs=n_jobs)
        # Dataset only for points. Just for testing.
        learning_point_dict = dict()
        for srcid, tagsets in chain(learning_truths_dict.items(),
                                    validation_truths_dict.items()):
            point_tagset = 'none'
            for tagset in tagsets:
                if tagset in point_tagsets:
                    point_tagset = tagset
                    break
            learning_point_dict[srcid] = point_tagset
        learning_point_dict['dummy'] = 'unknown'
        point_truths_dict = dict()
        point_srcids = list()
        for srcid in learning_srcids:
            truths = learning_truths_dict[srcid]
            point_tagset = None
            for tagset in truths:
                if tagset in point_tagsets:
                    point_tagset = tagset
                    break
            if point_tagset:
                point_truths_dict[srcid] = point_tagset
                point_srcids.append(srcid)

        try:
            point_truth_mat = [point_tagsets.index(point_truths_dict[srcid]) \
                               for srcid in point_srcids]
            point_vect_doc = np.vstack([learning_vect_doc[learning_srcids.index(srcid)]
                                        for srcid in point_srcids])
        except:
            pdb.set_trace()


    def _augment_with_ts(self, test_phrases_dict):
        # TODO: Implement below
        ts_learning_srcids = list()
        learning_tags_dict = {srcid: splitter(self.point_dict[srcid])
                              for srcid in self.learning_srcids}

        tag_binarizer = MultiLabelBinarizer()
        tag_binarizer.fit(map(splitter, self.point_dict.values()))
        with open(self.ts_feature_filename, 'rb') as fp:
            ts_features = pickle.load(fp, encoding='bytes')
        new_ts_features = list()
        for ts_feature in ts_features:
            feats = ts_feature[0]
            srcid = ts_feature[2]
            if srcid in self.learning_srcids + self.validation_srcids:
                point_tagset = self.point_dict[srcid]
                point_tags = point_tagset.split('_')
                point_vec = tag_binarizer.transform([point_tags])
                new_feature = [feats, point_vec, srcid]
                new_ts_features.append(new_feature)
            elif srcid in self.target_srcids:
                new_ts_features.append(ts_feature)
        ts_features = new_ts_features

        self.ts2ir = TimeSeriesToIR(mlb=tag_binarizer)
        self.ts2ir.fit(ts_features, self.learning_srcids, self.validation_srcids, learning_tags_dict)
        learning_ts_tags_pred = self.ts2ir.predict(ts_features, self.learning_srcids)
        for srcid, ts_tags in zip(self.learning_srcids, \
                                  tag_binarizer.inverse_transform(
                                      learning_ts_tags_pred)):
            #learning_phrase_dict[srcid] += list(ts_tags)
            ts_srcid = srcid + '_ts'
            learning_phrase_dict[ts_srcid] = learning_phrase_dict[srcid]\
                                                + list(ts_tags)
            ts_learning_srcids.append(ts_srcid)
            learning_truths_dict[ts_srcid] = learning_truths_dict[srcid]

        test_ts_tags_pred = self.ts2ir.predict(ts_features, test_srcids)
        for srcid, ts_tags in zip(test_srcids, \
                                  tag_binarizer.inverse_transform(
                                      test_ts_tags_pred)):
            #ts_srcid = srcid + '_ts'
            #test_phrase_dict[ts_srcid] = test_phrase_dict[srcid] + list(ts_tags)
            #test_srcids .append(ts_srcid) # TODO: Validate if this works.
            test_phrase_dict[srcid] += list(ts_tags)

    def _augment_negative_examples(self, doc, srcids):
        negative_doc = []
        negative_truths_dict = {}
        negative_srcids = []
        for srcid in self.learning_srcids:
            true_tagsets = list(set(self.tagsets_dict[srcid]))
            sentence = self.phrase_dict[srcid]
            for tagset in true_tagsets:
                negative_srcid = srcid + ';' + gen_uuid()
                removing_tagsets = set()
                new_removing_tagsets = set([tagset])
                removing_tags = []
                negative_tagsets = list(filter(tagset.__ne__, true_tagsets))
                i = 0
                while len(new_removing_tagsets) != len(removing_tagsets):
                    i += 1
                    if i>5:
                        pdb.set_trace()
                    removing_tagsets = deepcopy(new_removing_tagsets)
                    for removing_tagset in removing_tagsets:
                        removing_tags += removing_tagset.split('_')
                    for negative_tagset in negative_tagsets:
                        for tag in removing_tags:
                            if tag in negative_tagset.split('_'):
                                new_removing_tagsets.add(negative_tagset)
                negative_sentence = [tag for tag in sentence if\
                                     tag not in removing_tags]
                for tagset in removing_tagsets:
                    negative_tagsets = list(filter(tagset.__ne__,
                                                   negative_tagsets))

    #            negative_sentence = [word for word in sentence \
    #                                 if word not in tagset.split('_')]
                negative_doc.append(' '.join(negative_sentence))
                negative_truths_dict[negative_srcid] = negative_tagsets
                negative_srcids.append(negative_srcid)
        """
        for i in range(0, self.nb_empty_docs):
            # Add empty examples
            negative_srcid = gen_uuid()
            negative_doc.append('')
            negative_srcids.append(negative_srcid)
            negative_truths_dict[negative_srcid] = []
        """
        doc += negative_doc
        srcids += negative_srcids
        self.tagsets_dict.update(negative_truths_dict)
        return doc, srcids

    def _augment_brick_samples(self, doc, srcids):
        brick_truths_dict = dict()
        self.brick_srcids = []
        brick_doc = []
        logging.info('Start adding Brick samples')
        brick_copy_num = 6
        self.brick_tagsets_dict = dict()
        self.brick_doc = list()
        for tagset in self.tagset_list:
            for j in range(0, brick_copy_num):
                #multiplier = random.randint(2, 6)
                srcid = 'brick;' + gen_uuid()
                self.brick_srcids.append(srcid)
                self.brick_tagsets_dict[srcid] = [tagset]
                tagset_doc = list()
                for tag in tagset.split('_'):
                    tagset_doc += [tag] * random.randint(1,2)
                brick_doc.append(' '.join(tagset_doc))
        doc += brick_doc
        self.tagsets_dict.update(self.brick_tagsets_dict)
        srcids += self.brick_srcids
        return doc, srcids


    def _augment_eda(self):
        if eda_flag:
            unlabeled_phrase_dict = make_phrase_dict(\
                                        test_sentence_dict, \
                                        test_token_label_dict, \
                                        {target_building:test_srcids},\
                                        False)
            prefixer = build_prefixer(target_building)
            unlabeled_target_doc = [' '.join(\
                                    map(prefixer, unlabeled_phrase_dict[srcid]))\
                                    for srcid in test_srcids]
#        unlabeled_vect_doc = - tagset_vectorizer\
#                               .transform(unlabeled_target_doc)
            unlabeled_vect_doc = np.zeros((len(test_srcids), \
                                           len(tagset_vectorizer.vocabulary_)))
            target_doc = [' '.join(unlabeled_phrase_dict[srcid])\
                             for srcid in test_srcids]
            test_vect_doc = tagset_vectorizer.transform(target_doc).toarray()
            for building in source_target_buildings:
                if building == target_building:
                    added_test_vect_doc = - test_vect_doc
                else:
                    added_test_vect_doc = test_vect_doc
                unlabeled_vect_doc = np.hstack([unlabeled_vect_doc,\
                                                added_test_vect_doc])

        if eda_flag:
            learning_vect_doc = tagset_vectorizer.transform(learning_doc +
                                                            negative_doc).todense()
            learning_srcids += negative_srcids
            new_learning_vect_doc = deepcopy(learning_vect_doc)
            for building in source_target_buildings:
                building_mask = np.array([1 if find_key(srcid.split(';')[0],
                                                        total_srcid_dict,
                                                        check_in) == building.id
                                          else 0 for srcid in learning_srcids])
                new_learning_vect_doc = np.hstack([new_learning_vect_doc] \
                                     + [np.asmatrix(building_mask \
                                        * np.asarray(learning_vect)[0]).T \
                                    for learning_vect \
                                        in learning_vect_doc.T])
            learning_vect_doc = new_learning_vect_doc
            if use_brick_flag:
                new_brick_srcids = list()
                new_brick_vect_doc = np.array([])\
                        .reshape((0, len(tagset_vectorizer.vocabulary) \
                                  * (len(source_target_buildings)+1)))
                brick_vect_doc = tagset_vectorizer.transform(brick_doc).todense()
                for building in source_target_buildings:
                    prefixer = lambda srcid: building.id + '-' + srcid
                    one_brick_srcids = list(map(prefixer, brick_srcids))
                    for new_brick_srcid, brick_srcid\
                            in zip(one_brick_srcids, brick_srcids):
                        brick_truths_dict[new_brick_srcid] = \
                                brick_truths_dict[brick_srcid]
                    one_brick_vect_doc = deepcopy(brick_vect_doc)
                    for b in source_target_buildings:
                        if b != building:
                            one_brick_vect_doc = np.hstack([
                                one_brick_vect_doc,
                                np.zeros((len(brick_srcids),
                                          len(tagset_vectorizer.vocabulary)))])
                        else:
                            one_brick_vect_doc = np.hstack([
                                one_brick_vect_doc, brick_vect_doc])
                    new_brick_vect_doc = np.vstack([new_brick_vect_doc,
                                                one_brick_vect_doc])
                    new_brick_srcids += one_brick_srcids
                learning_vect_doc = np.vstack([learning_vect_doc,
                                               new_brick_vect_doc])
                brick_srcids = new_brick_srcids
                learning_srcids += brick_srcids

    def _build_tagset_classifier(self,
                                 learning_srcids,
                                 target_srcids,
                                 validation_srcids):

        learning_srcids = deepcopy(learning_srcids)

        # Update TagSet pool to include TagSets not in Brick.
        #orig_sample_num = len(learning_srcids)
        #new_tagset_list = tree_flatter(self.tagset_tree, [])
        #new_tagset_list = [tagset for tagset in new_tagset_list
        #                   if tagset not in ['location', 'equipment']]
        #TODO: fix tagset_tree instead of using the above temp fix.
        #new_tagset_list = new_tagset_list + [ts for ts in self.tagset_list \
        #                                     if ts not in new_tagset_list]
        #self.tagset_list = new_tagset_list
        self.tagset_binarizer = MultiLabelBinarizer(self.tagset_list)
        self.tagset_binarizer.fit([self.tagset_list])
        assert self.tagset_list == self.tagset_binarizer.classes_.tolist()

        #self.tagsets_dict = {srcid: self.tagsets_dict[srcid] 
        #                         for srcid in learning_srcids}


        ## Init brick tag_list
        # TODO: Maybe this should be done in initialization stage.
        self.tag_list = list(set(reduce(adder, map(splitter, 
                                                   self.tagset_list))))

        # All possible vocabularies.
        vocab_dict = dict([(tag, i) for i, tag in enumerate(self.tag_list)])

        # Define Vectorizer
        tokenizer = lambda x: x.split()
        # TODO: We could use word embedding like word2vec here instead.
        if self.vectorizer_type == 'tfidf':
            self.tagset_vectorizer = TfidfVectorizer(tokenizer=tokenizer, # TODO: This should be renamed as tags_vectorizer
                                                vocabulary=vocab_dict)
        elif self.vectorizer_type == 'meanbembedding':
            self.tagset_vectorizer = MeanEmbeddingVectorizer(tokenizer=tokenizer, 
                                                        vocabulary=vocab_dict)
        elif self.vectorizer_type == 'count':
            self.tagset_vectorizer = CountVectorizer(tokenizer=tokenizer,
                                                vocabulary=vocab_dict)
        else:
            raise Exception('Wrong vectorizer type: {0}'
                                .format(self.vectorizer_type))

        ## Transform learning samples
        if self.use_known_tags: #TODO: Remove this is not necessary
            learning_doc = [' '.join(self.phrase_dict[srcid] +
                                     self.known_tags_dict[srcid])
                            for srcid in learning_srcids]
            target_doc = [' '.join(self.phrase_dict[srcid] +
                                 self.known_tags_dict[srcid])
                        for srcid in target_srcids]
            learning_doc += [' '.join(self.phrase_dict[srcid])
                            for srcid in learning_srcids]
            target_doc += [' '.join(self.phrase_dict[srcid])
                        for srcid in target_srcids]
            learning_srcids *= 2
        else:
            learning_doc = [' '.join(self.phrase_dict[srcid]) for srcid in learning_srcids]
            #target_doc = [' '.join(self.phrase_dict[srcid]) for srcid in target_srcids]

        # Augment with negative examples.
        if self.negative_flag:
            learning_doc, learning_srcids = self._augment_negative_examples(learning_doc,
                                                                             learning_srcids)


        # Init Brick samples.
        if self.use_brick_flag:
            learning_doc, learning_srcids = self._augment_brick_samples(learning_doc,
                                                                         learning_srcids)

        # Init domain vector of source
        learning_domain_doc = [self.get_srcid_domain(srcid) for srcid in learning_srcids]
        # Add empty examples to each domain
        if self.emptydoc_flag:
            domain_types = set(learning_domain_doc)
            for domain_type in domain_types:
                for i in range(0, int(self.nb_empty_docs / len(domain_types))):
                    empty_srcid = gen_uuid()
                    learning_srcids.append(empty_srcid)
                    learning_domain_doc.append(domain_type)
                    learning_doc.append('')
                    self.tagsets_dict[empty_srcid] = []

        # Init domain vector of target
        target_domain_doc = [self.get_srcid_domain(srcid) for srcid in target_srcids]

        self.domain_vectorizer = CountVectorizer()
        self.domain_vectorizer.fit(learning_domain_doc + target_domain_doc)
        learning_domain_vect_doc = self.domain_vectorizer.transform(learning_domain_doc).todense()
        target_domain_vect_doc = self.domain_vectorizer.transform(target_domain_doc).todense()

        self.tagset_vectorizer.fit(learning_doc)#+ target_doc)# + brick_doc)

        # Apply Easy-Domain-Adaptation mechanism. Not useful.
        if self.eda_flag:
            raise Exception('Not implemented')
            # TODO: self._augment_eda()
        else:
            # Make TagSet vectors.

            learning_vect_doc = self.tagset_vectorizer.transform(learning_doc).todense()
            #target_vect_doc = self.tagset_vectorizer.transform(target_doc).todense()

        truth_mat = csr_matrix([self.tagset_binarizer.transform(
                                    [self.tagsets_dict[srcid]])[0]
                                for srcid in learning_srcids])
        if self.eda_flag:
            raise Exception('Not implemented')
            zero_vectors = self.tagset_binarizer.transform(\
                        [[] for i in range(0, unlabeled_vect_doc.shape[0])])
            truth_mat = vstack([truth_mat, zero_vectors])
            learning_vect_doc = np.vstack([learning_vect_doc, unlabeled_vect_doc])

        logging.info('Start learning multi-label classifier')
        ## Learn the classifier. StructuredCC is the default model.
        if self.tagset_classifier_type == 'RandomForest':
            def meta_rf(**kwargs):
                #return RandomForestClassifier(**kwargs)
                return RandomForestClassifier(n_jobs=self.n_jobs, n_estimators=150)
            meta_classifier = meta_rf
            params_list_dict = {}
        elif self.tagset_classifier_type == 'StructuredCC_BACKUP':
            #feature_selector = SelectFromModel(LinearSVC(C=0.001))
            feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
            base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
            #base_base_classifier = GradientBoostingClassifier()
            #base_base_classifier = RandomForestClassifier()
            base_classifier = Pipeline([('feature_selection',
                                         feature_selector),
                                        ('classification',
                                         base_base_classifier)
                                       ])
            tagset_classifier = StructuredClassifierChain(
                                    base_classifier,
                                    self.tagset_binarizer,
                                    subclass_dict,
                                    self.tagset_vectorizer.vocabulary,
                                    n_jobs,
                                    use_brick_flag)
        elif self.tagset_classifier_type == 'Project':
            def meta_proj(**kwargs):
                #base_classifier = LinearSVC(C=20, penalty='l1', dual=False)
                base_classifier = SVC(kernel='rbf', C=10, class_weight='balanced')
                #base_classifier = GaussianProcessClassifier()
                tagset_classifier = ProjectClassifier(base_classifier,
                                                               self.tagset_binarizer,
                                                               self.tagset_vectorizer,
                                                               subclass_dict,
                                                               n_jobs)
                return tagset_classifier
            meta_classifier = meta_proj
            params_list_dict = {}

        elif self.tagset_classifier_type == 'CC':
            def meta_cc(**kwargs):
                feature_selector = SelectFromModel(LinearSVC(C=1))
                #feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
                base_base_classifier = GradientBoostingClassifier(**kwargs)
                #base_base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet')
                #base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
                #base_base_classifier = LogisticRegression()
                #base_base_classifier = RandomForestClassifier(**kwargs)
                base_classifier = Pipeline([('feature_selection',
                                             feature_selector),
                                            ('classification',
                                             base_base_classifier)
                                           ])
                tagset_classifier = ClassifierChain(classifier=base_classifier)
                return tagset_classifier
            meta_classifier = meta_cc
            params_list_dict = {}

        elif self.tagset_classifier_type == 'StructuredCC_autoencoder':
            def meta_scc(**kwargs):
                feature_selector = SelectFromModel(LinearSVC(C=5))
                #feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
                base_base_classifier = GradientBoostingClassifier(**kwargs)
                #base_base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet')
                #base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
                #base_base_classifier = LogisticRegression()
                #base_base_classifier = RandomForestClassifier(**kwargs)
                base_classifier = Pipeline([('feature_selection',
                                             feature_selector),
                                            ('classification',
                                             base_base_classifier)
                                           ])
                tagset_classifier = StructuredClassifierChain(
                                    base_classifier,
                                    self.tagset_binarizer,
                                    self.subclass_dict,
                                    self.tagset_vectorizer.vocabulary,
                                    self.n_jobs,
                                    self.use_brick_flag,
                                    self.tagset_vectorizer)
                return tagset_classifier
            meta_classifier = meta_scc
            rf_params_list_dict = {
                'n_estimators': [10, 50, 100],
                'criterion': ['gini', 'entropy'],
                'max_features': [None, 'auto'],
                'max_depth': [1, 5, 10, 50],
                'min_samples_leaf': [2,4,8],
                'min_samples_split': [2,4,8]
            }
            gb_params_list_dict = {
                'loss': ['deviance', 'exponential'],
                'learning_rate': [0.1, 0.01, 1, 2],
                'criterion': ['friedman_mse', 'mse'],
                'max_features': [None, 'sqrt'],
                'max_depth': [1, 3, 5, 10],
                'min_samples_leaf': [1,2,4,8],
                'min_samples_split': [2,4,8]
            }
            params_list_dict = gb_params_list_dict

        elif self.tagset_classifier_type == 'StructuredCC':
            def meta_scc(**kwargs):
                #feature_selector = SelectFromModel(LinearSVC(C=5))
                #feature_selector = SelectFromModel(LinearSVC(C=1))
                feature_selector = SelectFromModel(LinearSVC(C=1))
                #feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
                base_base_classifier = GradientBoostingClassifier(**kwargs)
                #base_base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet')
                #base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
                #base_base_classifier = LogisticRegression()
                #base_base_classifier = RandomForestClassifier(**kwargs)
                base_classifier = Pipeline([('feature_selection',
                                             feature_selector),
                                            ('classification',
                                             base_base_classifier)
                                           ])
                tagset_classifier = StructuredClassifierChain(
                                    base_classifier,
                                    self.tagset_binarizer,
                                    self.subclass_dict,
                                    self.tagset_vectorizer.vocabulary,
                                    self.n_jobs,
                                    self.use_brick_flag,
                                    self.tagset_vectorizer)
                return tagset_classifier
            meta_classifier = meta_scc
            rf_params_list_dict = {
                'n_estimators': [10, 50, 100],
                'criterion': ['gini', 'entropy'],
                'max_features': [None, 'auto'],
                'max_depth': [1, 5, 10, 50],
                'min_samples_leaf': [2,4,8],
                'min_samples_split': [2,4,8]
            }
            gb_params_list_dict = {
                'loss': ['deviance', 'exponential'],
                'learning_rate': [0.1, 0.01, 1, 2],
                'criterion': ['friedman_mse', 'mse'],
                'max_features': [None, 'sqrt'],
                'max_depth': [1, 3, 5, 10],
                'min_samples_leaf': [1,2,4,8],
                'min_samples_split': [2,4,8]
            }
            params_list_dict = gb_params_list_dict
        elif self.tagset_classifier_type == 'StructuredCC_RF':
            base_classifier = RandomForestClassifier()
            tagset_classifier = StructuredClassifierChain(base_classifier,
                                                          self.tagset_binarizer,
                                                          subclass_dict,
                                                          self.tagset_vectorizer.vocabulary,
                                                          n_jobs)
        elif self.tagset_classifier_type == 'StructuredCC_LinearSVC':
            def meta_scc_svc(**kwargs):
                base_classifier = LinearSVC(loss='hinge', tol=1e-5,\
                                            max_iter=2000, C=2,
                                            fit_intercept=False,
                                            class_weight='balanced')
                tagset_classifier = StructuredClassifierChain(base_classifier,
                                                              self.tagset_binarizer,
                                                              subclass_dict,
                                                              self.tagset_vectorizer.vocabulary,
                                                              n_jobs)
                return tagset_classifier
            params_list_dict = {}
            meta_classifier = meta_scc_svc
        elif self.tagset_classifier_type == 'OneVsRest':
            base_classifier = LinearSVC(loss='hinge', tol=1e-5,\
                                        max_iter=2000, C=2,
                                        fit_intercept=False,
                                        class_weight='balanced')
            tagset_classifier = OneVsRestClassifier(base_classifier)
        elif self.tagset_classifier_type == 'Voting':
            def meta_voting(**kwargs):
                return VotingClassifier(self.tagset_binarizer, self.tagset_vectorizer,
                                        self.tagset_tree, self.tagset_list)
            meta_classifier = meta_voting
            params_list_dict = {}
        elif self.tagset_classifier_type == 'MLP':
            # Def model
            data_dim = learning_vect_doc.shape[1]
            output_classes = truth_mat.shape[1]
            model = self.get_mlp_model(data_dim, output_classes)
        elif self.tagset_classifier_type == 'DANN':
            data_dim = learning_vect_doc.shape[1]
            output_classes = truth_mat.shape[1]
            nb_domains = learning_domain_vect_doc.shape[-1]
            dann = DANN(data_dim, output_classes, nb_domains,
                        batch_size=128,
                        )
        else:
            raise Exception('Wrong tagset classifier type: {0}'
                            .format(self.tagset_classifier_type))

        if not isinstance(truth_mat, csr_matrix):
            truth_mat = csr_matrix(truth_mat)

        # TODO: Hyper-parameter optimization. (But expect it'd be slow.)
        if self.tagset_classifier_type == 'MLP':
            self.tagset_classifier = model
        elif self.tagset_classifier_type == 'DANN':
            self.tagset_classifier = dann
        else:
            best_params = {'learning_rate':0.1,
                           'subsample':0.25,
                           'n_estimators': 200}
            self.tagset_classifier  = meta_classifier(**best_params)


        # add an empty doc.
        #empty_doc_num = min(5, int(learning_vect_doc.shape[0]*0.02))
        #learning_vect_doc = np.hstack([learning_vect_doc,
        #                               np.zeros((empty_doc_num, learning_vect_doc.shape[1]))
        #                               ])

        # Actual fitting.
        if isinstance(self.tagset_classifier, StructuredClassifierChain):
            self.tagset_classifier.fit(learning_vect_doc, truth_mat.toarray(), \
                                  orig_sample_num=len(learning_vect_doc)
                                  - len(self.brick_srcids))
        elif self.tagset_classifier_type == 'MLP':
            self.tagset_classifier.fit(learning_vect_doc,
                                       truth_mat,
                                       batch_size=128,
                                       epochs=self.epochs,
                                       verbose=True)
        elif self.tagset_classifier_type == 'DANN':
            truth_mat = truth_mat.todense()
            target_domain_index = self.domain_vectorizer.vocabulary_[self.target_building.id]
            self.tagset_classifier.fit(learning_vect_doc, truth_mat, learning_domain_vect_doc,
                                       target_vect_doc, target_domain_vect_doc, target_domain_index,
                                       nb_epochs=1500,
                                       )
        else:
            self.tagset_classifier.fit(learning_vect_doc, truth_mat.toarray())
        #self.point_classifier.fit(point_vect_doc, point_truth_mat)
        logging.info('Finished learning multi-label classifier')

    def get_mlp_model(self, data_dim, output_classes):
        model = Sequential()
        model.add(Dense(64,
                        input_shape=(data_dim,),
                        #bias_regularizer=regularizers.l1(0.0001),
                        #kernel_regularizer=regularizers.l1(0.001),
                        #activity_regularizer=regularizers.l1(0.001),
                        #kernel_constraint=max_norm(3),
                        activation='relu'))
        """
        model.add(Dropout(0.1))
        model.add(Dense(64,
                        input_shape=(data_dim,),
                        #bias_regularizer=regularizers.l1(0.0001),
                        #kernel_regularizer=regularizers.l1(0.001),
                        #activity_regularizer=regularizers.l1(0.001),
                        kernel_constraint=max_norm(3),
                        activation='relu'))
        """
        model.add(Dropout(0.1))
        model.add(Dense(output_classes,
                        #bias_regularizer=regularizers.l1(0.0001),
                        #kernel_regularizer=regularizers.l1(0.0001),
                        #activity_regularizer=regularizers.l2(0.01),
                        #kernel_constraint=max_norm(3),
                        activation='sigmoid'))
        #model.compile(optimizer='sgd',
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      )
        return model

    def _parameter_validation(vect_doc, truth_mat, srcids, params_list_dict,
                             meta_classifier, vectorizer, binarizer,
                             source_target_buildings, eda_flag):
        # TODO: This is not effective for now. Do I need one?
        #best_params = {'n_estimators': 50, 'criterion': 'entropy', 'max_features': 'auto', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}
        #best_params = {'criterion': 'entropy'}
        #best_params = {'loss': 'exponential', 'learning_rate': 0.01, 'criterion': 'friedman_mse', 'max_features': None, 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}

            #tagset_classifier = RandomForestClassifier(n_estimators=100,
            #                                           random_state=0,\
            #                                           n_jobs=n_jobs)
        best_params = {'learning_rate':0.1, 'subsample':0.25}
        #best_params = {'C':0.4, 'solver': 'liblinear'}

        return meta_classifier(**best_params) # Pre defined setup.

        #best_params = {'n_estimators': 120, 'n_jobs':7}
        #return meta_classifier(**best_params)

        token_type = 'justseparate'
        results_dict = dict()
        for key, values in params_list_dict.items():
            results_dict[key] = {'ha': [0]*len(values),
                                 'a': [0]*len(values),
                                 'mf1': [0]*len(values)}
        avg_num = 3
        for i in range(0,avg_num):
            learning_indices = random.sample(range(0, len(srcids)),
                                             int(len(srcids)/2))
            validation_indices = [i for i in range(0, len(srcids))
                                  if i not in learning_indices]
            learning_srcids = [srcids[i] for i
                                        in learning_indices]
            validation_srcids = [srcids[i] for i
                                 in validation_indices]
            for key, values in params_list_dict.items():
                for j, value in enumerate(values):
                    params = {key: value}
                    classifier = meta_classifier(**params)
                    classifier.fit(vect_doc[learning_indices], \
                                   truth_mat[learning_indices].toarray())

                    validation_sentence_dict, \
                    validation_token_label_dict, \
                    validation_truths_dict, \
                    validation_phrase_dict = self.get_multi_buildings_data(\
                                                source_target_buildings, validation_srcids, \
                                                eda_flag, token_type)

                    validation_pred_tagsets_dict, \
                    validation_pred_certainty_dict, \
                    _ = tagsets_prediction(classifier, vectorizer, binarizer, \
                                       validation_phrase_dict, validation_srcids, \
                                       source_target_buildings, eda_flag, None,
                                           ts2ir=None)
                    validation_result = tagsets_evaluation(validation_truths_dict, \
                                                           validation_pred_tagsets_dict, \
                                                           validation_pred_certainty_dict,\
                                                           validation_srcids, \
                                                           None, \
                                                           validation_phrase_dict, \
                                                           debug_flag=False,
                                                           classifier=classifier, \
                                                           vectorizer=vectorizer)
                    results_dict[key]['ha'][j] += validation_result['hierarchy_accuracy']
                    results_dict[key]['a'][j] += validation_result['accuracy']
                    results_dict[key]['mf1'][j] += validation_result['macro_f1']
                    results_dict[key]['macro_f1'][j] += validation_result['macro_f1']
        best_params = dict()
        for key, results in results_dict.items():
            metrics = results_dict[key]['mf1']
            best_params[key] = params_list_dict[key][metrics.index(max(metrics))]
        classifier = meta_classifier(**best_params)
        classifier.fit(vect_doc[learning_indices], \
                       truth_mat[learning_indices].toarray())

        validation_sentence_dict, \
        validation_token_label_dict, \
        validation_truths_dict, \
        validation_phrase_dict = self.get_multi_buildings_data(\
                                    source_target_buildings, validation_srcids, \
                                    eda_flag, token_type)

        validation_pred_tagsets_dict, \
        validation_pred_certainty_dict, \
        _ = tagsets_prediction(classifier, vectorizer, binarizer, \
                           validation_phrase_dict, validation_srcids, \
                           source_target_buildings, eda_flag, None,
                               ts2ir=None)
        validation_result = tagsets_evaluation(validation_truths_dict, \
                                               validation_pred_tagsets_dict, \
                                               validation_pred_certainty_dict,\
                                               validation_srcids, \
                                               None, \
                                               validation_phrase_dict, \
                                               debug_flag=False,
                                               classifier=classifier, \
                                               vectorizer=vectorizer)
        best_ha = validation_result['hierarchy_accuracy']
        best_a = validation_result['accuracy']
        best_mf1 = validation_result['macro_f1']

        return meta_classifier(**best_params)

    def update_phrases(self, phrases):
        self.phrase_dict.update(phrases)

