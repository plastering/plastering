import os
from uuid import uuid4
from operator import itemgetter
from pathlib import Path

import pycrfsuite
from bson.binary import Binary as BsonBinary
import arrow
import numpy as np
import pandas as pd

from mongo_models import store_model, get_model, get_tags_mapping, \
    get_crf_results, store_result, get_entity_results
from base_scrabble import BaseScrabble
from common import *

curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))

def gen_uuid():
    return str(uuid4())


class Ir2Entities(BaseScrabble):
    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 source_buildings=[],
                 source_sample_num_list=[],
                 learning_srcids=[],
                 conf={}
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
                 conf)
        self.model_uuid = None

        if 'crftype' in conf:
            self.crftype = conf['crftype']
        else:
            self.crftype = 'crfsuite'
        if 'query_strategy' in conf:
            self.query_strategy = conf['query_strategy']
        else:
            self.query_strategy = 'confidence'
        if 'user_cluster_flag' in conf:
            self.use_cluster_flag = conf['use_cluster_flag']
        else:
            self.use_cluster_flag = True

        # Note: Hardcode to disable use_brick_flag
        """
        if 'use_brick_flag' in conf:
            self.use_brick_flag = conf['use_brick_flag']
        else:
            self.use_brick_flag = False  # Temporarily disable it
        """
        self.use_brick_flag = False
        self._init_data()

    def _init_data(self):
        self.sentence_dict = {}
        self.label_dict = {}
        for building, source_sample_num in zip(self.source_buildings,
                                               self.source_sample_num_list):
            self.sentence_dict.update(self.building_sentence_dict[building])
            one_label_dict = self.building_label_dict[building]
            self.label_dict.update(one_label_dict)

            if not self.learning_srcids:
                sample_srcid_list = select_random_samples(
                                        building,
                                        one_label_dict.keys(),
                                        source_sample_num,
                                        self.use_cluster_flag)
                self.learning_srcids += sample_srcid_list

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
        self.brick_srcids = list(brick_sentence_dict.keys())

    def update_model(self, srcids):
        assert (len(self.source_buildings) == len(self.source_sample_num_list))
        self.learning_srcids += srcids

        algo = 'ap'
        trainer = pycrfsuite.Trainer(verbose=False, algorithm=algo)
        if algo == 'ap':
            trainer.set('max_iterations', 200)

            # algorithm: {'lbfgs', 'l2sgd', 'ap', 'pa', 'arow'}
        trainer.set_params({'feature.possible_states': True,
                            'feature.possible_transitions': True})
        for srcid in self.learning_srcids:
            sentence = self.sentence_dict[srcid]
            labels = self.label_dict[srcid]
            trainer.append(pycrfsuite.ItemSequence(
                self._calc_features(sentence, None)), labels)
        if self.use_brick_flag:
            for srcid in self.brick_srcids:
                sentence = self.brick_sentence_dict[srcid]
                labels = self.brick_label_dict[srcid]
                trainer.append(pycrfsuite.ItemSequence(
                    self._calc_features(sentence, None)), labels)
        model_uuid = gen_uuid()
        crf_model_file = 'temp/{0}.{1}.model'.format(model_uuid, 'crfsuite')
        trainer.train(crf_model_file)
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
        predicted_dict, score_dict = self._predict_func(model,
                                                        target_sentence_dict,
                                                        self.crftype)
        cluster_dict = get_cluster_dict(self.target_building)

        new_srcids = []
        if self.query_strategy == 'confidence':
            for srcid, score in score_dict.items():
                # Normalize with length
                score_dict[srcid] = np.log(score) / \
                                    len(self.sentence_dict[srcid])
            sorted_scores = sorted(score_dict.items(), key=itemgetter(1))

            # load word clusters not to select too similar samples.
            added_cids = []
            new_srcid_cnt = 0
            for srcid, score in sorted_scores:
                if srcid in self.target_srcids:
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
            if i in [0, 1]:
                features['SECOND'] = 1.0
            else:
                features['-2:word.lower=' + sentence[i - 2].lower()] = 1.0
            # if i<len(sentence)-1:
            #    features['+1:word.lower='+sentence[i+1].lower()] = 1.0
            # else:
            #    features['EOS'] = 1.0
            sentenceFeatures.append(features)
        return sentenceFeatures

    def _predict_func(self, model, sentence_dict, crftype):
        crf_model_file = 'temp/{0}.{1}.model'.format(self.model_uuid, crftype)
        """
        with open(crf_model_file, 'wb') as fp:
            fp.write(model['model_binary'])
        if crftype == 'crfsharp':
            for postfix in crfsharp_other_postfixes:
                with open(crf_model_file + postfix, 'wb') as fp:
                    fp.write(model['model_binary' + postfix.replace('.','_')])
        """
        self._load_crf_model_files(model, crf_model_file, crftype)

        predicted_dict = dict()
        score_dict = dict()
        begin_time = arrow.get()
        if crftype == 'crfsuite':
            # Init tagger
            tagger = pycrfsuite.Tagger()
            tagger.open(crf_model_file)

            # Tagging sentences with tagger
            for srcid, sentence in sentence_dict.items():
                predicted = tagger.tag(self._calc_features(sentence))
                predicted_dict[srcid] = predicted
                score_dict[srcid] = tagger.probability(predicted)
        elif crftype == 'crfsharp':
            tagger = CRFSharp(base_dir='./temp',
                              template='./model/scrabble.template',
                              thread=thread_num,
                              nbest=1,
                              modelfile=crf_model_file,
                              maxiter=crfsharp_maxiter
                              )
            srcids = list(sentence_dict.keys())
            sentences = [sentence_dict[srcid] for srcid in srcids]
            res = tagger.decode(sentences, srcids)
            for srcid in srcids:
                best_cand = res[srcid]['cands'][0]
                predicted_dict[srcid] = best_cand['token_predict']
                score_dict[srcid] = best_cand['prop']
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
