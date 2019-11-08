import os
from uuid import uuid4
from operator import itemgetter
from pathlib import Path
from collections import OrderedDict

import pycrfsuite
from bson.binary import Binary as BsonBinary
import arrow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, f1_score

from .mongo_models import store_model, get_model, get_tags_mapping, \
    get_crf_results, store_result, get_entity_results
from .base_scrabble import BaseScrabble
from .common import *
from . import eval_func

curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def gen_uuid():
    return str(uuid4())


class Char2Ir(BaseScrabble):
    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 source_buildings=[],
                 source_sample_num_list=[],
                 learning_srcids=[],
                 pgid=None,
                 config={}
                 ):
        super(Char2Ir, self).__init__(
            target_building,
            target_srcids,
            building_label_dict,
            building_sentence_dict,
            {},
            source_buildings,
            source_sample_num_list,
            learning_srcids,
            pgid,
            config)
        self.model_uuid = None

        if 'crftype' in config:
            self.crftype = config['crftype']
        else:
            self.crftype = 'crfsuite'
        if 'crfalgo' in config:
            self.crfalgo = config['crfalgo']
        else:
            self.crfalgo = 'ap'
        if 'crfqs' in config:
            self.query_strategy = config['crfqs']
        else:
            self.query_strategy = 'confidence'
        if 'user_cluster_flag' in config:
            self.use_cluster_flag = config['use_cluster_flag']
        else:
            self.use_cluster_flag = True

        if 'available_metadata_types' in config:
            self.available_metadata_types = config['available_metadata_types']
        else:
            self.available_metadata_types = ['VendorGivenName',
                                             'BACnetDescription',
                                             'BACnetName',
                                             ]

        if 'concatenate_sentences' in config:
            self.concatenate_sentences = config['concatenate_sentences']
        else:
            self.concatenate_sentences = False

        # Note: Hardcode to disable use_brick_flag
        """
        if 'use_brick_flag' in config:
            self.use_brick_flag = config['use_brick_flag']
        else:
            self.use_brick_flag = False  # Temporarily disable it
        """
        self.use_brick_flag = False
        self._init_data(learning_srcids)

    def order_sentence_dict(self, sentence_dict):
        return {srcid: OrderedDict(sentences)
                for srcid, sentences in sentence_dict.items()}


    def _init_data(self, learning_srcids=[]):
        self.sentence_dict = {}
        self.label_dict = {}
        self.building_cluster_dict = {}
        self.degrade_mask = [] # only used by char2ir_gpu.py
        for building, source_sample_num in zip(self.source_buildings, self.source_sample_num_list):
            self.sentence_dict.update(self.building_sentence_dict[building.id])
            one_label_dict = self.building_label_dict[building.id]
            self.label_dict.update(one_label_dict)

            if learning_srcids:
                self.learning_srcids = learning_srcids
                curr_sample_len = len(learning_srcids)
            else:
                sample_srcid_list = select_random_samples(
                    building=building,
                    srcids=one_label_dict.keys(),
                    n=source_sample_num,
                    use_cluster_flag=self.use_cluster_flag,
                    sentence_dict=self.building_sentence_dict[building.id],
                    shuffle_flag=False,
                )
                self.learning_srcids += sample_srcid_list
                curr_sample_len = len(sample_srcid_list)
            if building == self.target_building:
                self.degrade_mask += [0] * curr_sample_len
            else:
                self.degrade_mask += [1] * curr_sample_len
            if building not in self.building_cluster_dict:
                self.building_cluster_dict[building.id] = get_word_clusters(
                    self.building_sentence_dict[building.id])

        # Construct Brick examples
        brick_sentence_dict = dict()
        brick_label_dict = dict()
        if self.use_brick_flag:
            with open(curr_dir / 'metadata/brick_tags_labels.json', 'r') as fp:
                tag_label_list = json.load(fp)
            for tag_labels in tag_label_list:
                # Append meaningless characters before and after the tag
                # to make it separate from dependencies.
                # But comment them out to check if it works.
                # char_tags = [' '] + list(map(itemgetter(0), tag_labels)) + [' ']
                char_tags = list(map(itemgetter(0), tag_labels))
                # char_labels = ['O'] + list(map(itemgetter(1), tag_labels)) + ['O']
                char_labels = list(map(itemgetter(1), tag_labels))
                brick_sentence_dict[''.join(char_tags)] = char_tags + ['NEWLINE']
                brick_label_dict[''.join(char_tags)] = char_labels + ['O']
            self.sentence_dict.update(brick_sentence_dict)
            self.label_dict.update(brick_label_dict)
        self.sentence_dict = self.order_sentence_dict(self.sentence_dict)
        self.brick_srcids = list(brick_sentence_dict.keys())

    def merge_sentences(self, sentences):
        return '@\t@'.join(['@'.join(sentences[column]) for column in column_names
                            if column in sentences]).split('@')
    def merge_labels(self, labels):
        return '@O@'.join(['@'.join(labels[column]) for column in column_names
                           if column in labels]).split('@')


    def _add_point_to_model(self, srcid, trainer):
        if self.concatenate_sentences:
            sentence = self.merge_sentences(self.sentence_dict[srcid])
            labels = self.merge_labels(self.label_dict[srcid])
            assert len(sentence) == len(labels)
            trainer.append(pycrfsuite.ItemSequence(
                self._calc_features(sentence, None)), labels)
        else:
            for metadata_type, sentence in self.sentence_dict[srcid].items():
                #labels = self.label_dict[srcid][metadata_type]
                labels = self.get_fullparsing_labels(srcid, metadata_type)
                if len(labels) != len(sentence):
                    pdb.set_trace()
                trainer.append(pycrfsuite.ItemSequence(
                    self._calc_features(sentence, None)), labels)

    def update_model(self, srcids):
        assert (len(self.source_buildings) == len(self.source_sample_num_list))
        self.learning_srcids += srcids

        if self.crfalgo == 'prev':
            crfalgo = 'ap'
        else:
            crfalgo = self.crfalgo
        if self.crfalgo == 'default':
            trainer = pycrfsuite.Trainer(verbose=False)
        else:
            trainer = pycrfsuite.Trainer(verbose=False, algorithm=crfalgo)
        if self.crfalgo == 'ap':
            #trainer.set('max_iterations', 125)
            trainer.set('max_iterations', 200)
        elif self.crfalgo == 'lbfgs':
            trainer.set('c2', 0.02)

            # algorithm: {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
        if self.crfalgo != 'prev':
            trainer.set_params({'feature.possible_states': True,
                                'feature.possible_transitions': True})
        for srcid in self.learning_srcids:
            self._add_point_to_model(srcid, trainer)
        if self.use_brick_flag:
            for srcid in self.brick_srcids:
                sentence = self.brick_sentence_dict[srcid]
                labels = self.brick_label_dict[srcid]
                trainer.append(pycrfsuite.ItemSequence(
                    self._calc_features(sentence, None)), labels)
        model_uuid = gen_uuid()
        crf_model_file = 'temp/{0}.{1}.model'.format(model_uuid, 'crfsuite')
        t0 = arrow.get()
        trainer.train(crf_model_file)
        t1 = arrow.get()
        print('training crf took: {0}'.format(t1 - t0))
        with open(crf_model_file, 'rb') as fp:
            model_bin = fp.read()
        model = {
            # 'source_list': sample_dict,
            'gen_time': arrow.get().datetime,
            'use_cluster_flag': self.use_cluster_flag,
            'use_brick_flag': self.use_brick_flag,
            'model_binary': BsonBinary(model_bin),
            'source_building_count': len(self.source_buildings),
            'learning_srcids': sorted(set(self.learning_srcids)),
            'uuid': model_uuid,
            'crftype': 'crfsuite'
        }
        store_model(model)
        os.remove(crf_model_file)
        self.model_uuid = model_uuid

    @staticmethod
    def _get_model(model_uuid):
        model_query = {
            'uuid': model_uuid
        }
        model = get_model(model_query)
        return model

    def select_informative_samples(self, sample_num):
        target_sentence_dict = {srcid: self.sentence_dict[srcid]
                                for srcid in self.target_srcids}

        model = self._get_model(self.model_uuid)
        predicted_dict, score_dict, _ = self._predict_and_proba(self.target_srcids)
        # TODO: Validate if the above is same as before
        #predicted_dict, score_dict = self._predict_func(model,
        #                                                target_sentence_dict,
        #                                                self.crftype)
        cluster_dict = self.building_cluster_dict[self.target_building.id]

        new_srcids = []
        if self.query_strategy == 'confidence':
            for srcid, scores in score_dict.items():
                # Normalize with length
                curr_score = 0
                sentence_len = 0
                for metadata_type, score in scores.items():
                    sentence = self.sentence_dict[srcid][metadata_type]
                    if not sentence:
                        continue
                    curr_score += np.log(score)
                    sentence_len += len(sentence)
                score_dict[srcid] = curr_score / sentence_len
            sorted_scores = sorted(score_dict.items(), key=itemgetter(1))

            # load word clusters not to select too similar samples.
            added_cids = []
            new_srcid_cnt = 0
            for srcid, score in sorted_scores:
                if srcid in self.target_srcids:
                    if srcid in self.learning_srcids:
                        continue
                    the_cid = None
                    for cid, cluster in cluster_dict.items():
                        if srcid in cluster:
                            the_cid = cid
                            break
                    if the_cid in added_cids:
                        continue
                    added_cids.append(the_cid)
                    new_srcids.append(srcid)
                    new_srcid_cnt += 1
                    if new_srcid_cnt == sample_num:
                        break
        return new_srcids

    def _load_crf_model_files(self, model, filename, crftype):
        crf_model_file = filename
        with open(crf_model_file, 'wb') as fp:
            fp.write(model['model_binary'])

    def _calc_features(self, sentence, building=None):
        sentenceFeatures = list()
        sentence = ['$' if c.isdigit() else c for c in sentence]
        for i, word in enumerate(sentence):
            features = {
                'word.lower=' + word.lower(): 1.0,
                'word.isdigit': float(word.isdigit())
            }
            if i == 0:
                features['BOS'] = 1.0
            else:
                features['-1:word.lower=' + sentence[i - 1].lower()] = 1.0

            if i  == 0:
                pass
            elif i  == 1:
                features['SECOND'] = 1.0
            else:
                features['-2:word.lower=' + sentence[i - 2].lower()] = 1.0

            if i<len(sentence)-1:
                features['+1:word.lower='+sentence[i+1].lower()] = 1.0
            else:
                features['EOS'] = 1.0
            sentenceFeatures.append(features)
        return sentenceFeatures

    def divide_list(self, l, sep_indices):
        base_idx = 0
        d = defaultdict(list)
        column_idx = -1
        for column_idx, sep_idx in enumerate(sep_indices):
            curr_metadata_type = column_names[column_idx]
            for c in l[base_idx:sep_idx]:
                d[curr_metadata_type].append(c)
            base_idx = sep_idx
        if base_idx < len(l):
            curr_metadata_type = column_names[column_idx+1]
            for c in l[base_idx:]:
                d[curr_metadata_type].append(c)
        return dict(d)



    def _predict_func(self, model, sentence_dict, crftype):
        crf_model_file = 'temp/{0}.{1}.model'.format(self.model_uuid, crftype)
        self._load_crf_model_files(model, crf_model_file, crftype)

        predicted_dict = dict()
        score_dict = dict()
        begin_time = arrow.get()
        if crftype == 'crfsuite':
            # Init tagger
            tagger = pycrfsuite.Tagger()
            tagger.open(crf_model_file)

            # Tagging sentences with tagger
            for srcid, sentences in sentence_dict.items():
                predicteds = {}
                scores = {}
                if self.concatenate_sentences:
                    sentence = self.merge_sentences(sentences)
                    predicted = tagger.tag(self._calc_features(sentence))
                    score = tagger.probability(predicted)
                    predicteds['VendorGivenName'] = predicted
                    scores['VendorGivenName'] = score
                else:
                    for metadata_type, sentence in sentences.items():
                        predicted = tagger.tag(self._calc_features(sentence))
                        score = tagger.probability(predicted)
                        predicteds[metadata_type] = predicted
                        scores[metadata_type] = score
                predicted_dict[srcid] = predicteds
                score_dict[srcid] = scores
        return predicted_dict, score_dict

    def _predict_and_proba(self, target_srcids):
        # Validate if we have all information
        for srcid in target_srcids:
            try:
                assert srcid in self.sentence_dict
            except:
                pdb.set_trace()

        target_sentence_dict = {srcid: self.sentence_dict[srcid]
                                for srcid in target_srcids}
        model = self._get_model(self.model_uuid)
        predicted_dict, score_dict = self._predict_func(model,
                                                        target_sentence_dict,
                                                        self.crftype)
        # Construct output data
        pred_phrase_dict = make_phrase_dict(target_sentence_dict, predicted_dict)
        return predicted_dict, score_dict, pred_phrase_dict

    def predict(self, target_srcids=None):
        if not target_srcids:
            target_srcids = self.target_srcids
        predicted_dict, _, _ = self._predict_and_proba(target_srcids)
        return predicted_dict

    def predict_proba(self, target_srcids=None):
        if not target_srcids:
            target_srcids = self.target_srcids
        _, score_dict, _ = self._predict_and_proba(target_srcids)
        return score_dict

    def learn_auto(self, iter_num=1):
        pass

    def evaluate(self, preds):
        srcids = list(preds.keys())
        pred_tags_list = [reduce(adder,
                                 [preds[srcid][t]
                                  for t in self.available_metadata_types])
                          for srcid in srcids]
        true_tags_list = [reduce(adder,
                                 [self.label_dict[srcid][t]
                                  for t in self.available_metadata_types])
                          for srcid in srcids]
        acc = eval_func.sequential_accuracy(true_tags_list,
                                            pred_tags_list)

        pred = [preds[srcid] for srcid in preds.keys()]
        true = [self.label_dict[srcid] for srcid in preds.keys()]
        mlb = MultiLabelBinarizer()
        mlb.fit(pred + true)
        encoded_true = mlb.transform(true)
        encoded_pred = mlb.transform(pred)
        macro_f1 = f1_score(encoded_true, encoded_pred, average='macro')
        f1 = f1_score(encoded_true, encoded_pred, average='weighted')
        res = {
            'accuracy': acc,
            'f1': f1,
            'macro_f1': macro_f1
        }
        return res

