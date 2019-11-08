import os
import sys
import importlib.util
import pdb
from copy import deepcopy

from . import Inferencer
from ..common import POINT_TAGSET, ALL_TAGSETS, FULL_PARSING
from ..common import select_point_tagset, is_point_tagset
from .scrabble_helper import load_data
from .scrabble.scrabble import Scrabble  # This may imply incompatible imports.
from .scrabble.common import select_random_samples


@Inferencer()
class ScrabbleInterface(object):
    """docstring for ScrabbleInterface"""
    def __init__(self,
                 target_building,
                 target_srcids,
                 source_buildings,
                 config={},
                 **kwargs
                 ):
        config['required_label_types'] = [POINT_TAGSET,
                                          FULL_PARSING,
                                          ALL_TAGSETS]
        self.target_label_type = ALL_TAGSETS

        # Prepare config for Scrabble object
        if 'seed_num' in config:
            seed_num = config['seed_num']
        else:
            seed_num = 10

        if 'sample_num_list' in config:
            sample_num_list = config['sample_num_list']
        else:
            sample_num_list = [seed_num] * len(set(source_buildings))

        if self.target_building not in self.source_buildings:
            self.source_buildings = self.source_buildings + [self.target_building]
        if len(self.source_buildings) > len(sample_num_list):
            sample_num_list.append(0)

        if 'use_cluster_flag' not in config:
            config['use_cluster_flag'] = True
        if 'use_brick_flag' not in config:
            config['use_brick_flag'] = True
        if 'negative_flag' not in config:
            config['negative_flag'] = True
        if 'tagset_classifier_type' not in config:
            config['tagset_classifier_type'] = 'MLP'
        if 'crfqs' not in config:
            config['crfqs'] = 'confidence'
        if 'entqs' not in config:
            config['entqs'] = 'phrase_util'
        if 'n_jobs' not in config:
            config['n_jobs'] = 10
        if 'use_known_tags' not in config:
            config['use_known_tags'] = False
        if 'apply_filter_flag' not in config:
            self.apply_filter_flag = False
        else:
            self.apply_filter_flag = config['apply_filter_flag']
        if 'apply_validating_samples' not in config:
            self.apply_validating_samples = False
        else:
            self.apply_validating_samples = config['apply_validating_samples']

        # TODO: This should be migrated into Plastering
        building_sentence_dict, target_srcids, building_label_dict,\
            building_tagsets_dict, known_tags_dict = load_data(target_building,
                                                               self.source_buildings,
                                                               metadata_types=self.valid_metadata_types,
                                                               #metadata_types=['VendorGivenName',
                                                               #                'BACnetName',
                                                               #                'BACnetDescription',
                                                               #                'BACnetUnit',
                                                               #                ],
                                                               )
        self.scrabble = Scrabble(target_building,
                                 target_srcids,
                                 building_label_dict,
                                 building_sentence_dict,
                                 building_tagsets_dict,
                                 source_buildings,
                                 sample_num_list,
                                 known_tags_dict,
                                 config=config,
                                 pgid=self.pgid,
                                 )
        # new_srcids = deepcopy(self.scrabble.learning_srcids)
        if self.hotstart:
            new_srcids = [obj.srcid for obj in self.query_labels(building=target_building)]
            self.update_model(new_srcids)
        self.zodiac_good_preds = {}

    def learn_auto(self, iter_num=25, inc_num=10):
        for i in range(0, iter_num):
            print('--------------------------')
            print('{0}th iteration'.format(i))
            new_srcids = self.select_informative_samples(inc_num)
            self.update_model(new_srcids)
            self.evaluate(self.target_srcids)
            print('curr new srcids: {0}'.format(len(new_srcids)))
            print('training srcids: {0}'.format(len(self.training_srcids)))
            print('f1: {0}'.format(self.history[-1]['metrics']['f1']))
            print('macrof1: {0}'.format(self.history[-1]['metrics']['macrof1']))

    def update_model(self, new_srcids):
        self.scrabble.update_model(new_srcids)

    def postprocessing_pred(self, pred):
        # Currently only ingest point tagsets.
        pred_g = self.new_graph(empty=True)
        for srcid, tagsets in pred.items():
            point_tagset = select_point_tagset(tagsets, srcid)
            pred_g.add_pred_point_result(srcid, point_tagset)
        return pred_g

    def predict(self, target_srcids=None, output_format='ttl'):
        if not target_srcids:
            target_srcids = self.target_srcids
        pred = self.scrabble.predict(target_srcids)
        if self.apply_filter_flag:
            pred = self.apply_filter_by_zodiac(pred)
        self.pred_g = self.postprocessing_pred(pred)

        if output_format == 'ttl':
            return self.pred_g
        elif output_format == 'json':
            return pred

    def predict_proba(self, target_srcids=None):
        return self.scrabble.predict_proba(target_srcids)

    def apply_prior_zodiac(self, sample_num):
        if not self.prior_g:
            return []
        instances = self.prior_g.get_instance_tuples()
        good_preds = {}
        for srcid, point_tagset in instances.items():
            triple = (BASE[srcid], RDF.type, BRICK[point_tagset])
            if self.prior_confidences[triple] > 0.9:
                good_preds[srcid] = point_tagset
        pred_g = self.predict()
        incorrect_srcids = []
        for srcid, good_point_tagset in good_preds.items():
            pred_point_tagset = get_point_type(pred_g, BASE[srcid])
            if (good_point_tagset != pred_point_tagset) or\
               (good_point_tagset == 'unknown' and pred_point_tagset == 'none') or\
               (good_point_tagset == 'none' and pred_point_tagset == 'unknown'):
                incorrect_srcids.append(srcid)
        if not incorrect_srcids:
            return []
        new_srcids = select_random_samples(
            building=self.target_building,
            srcids=incorrect_srcids,
            n=sample_num,
            use_cluster_flag=True,
            sentence_dict=self.scrabble.char2ir.sentence_dict,
            unique_clusters_flag=True,
        )
        return new_srcids

    def is_same_tagset(self, tagset1, tagset2):
        if tagset1 == tagset2:
            return True
        elif tagset1 == 'none' and tagset2 == 'unknown':
            return True
        elif tagset1 == 'unknown' and tagset2 == 'none':
            return True
        else:
            return False

    def apply_filter_by_zodiac(self, pred):
        if not self.prior_g:
            return pred
        instances = self.prior_g.get_instance_tuples()
        self.zodiac_good_preds = {}
        for srcid, point_tagset in instances.items():
            triple = (BASE[srcid], RDF.type, BRICK[point_tagset])
            if self.prior_confidences[triple] > 0.8:
                self.zodiac_good_preds[srcid] = point_tagset
        fixed_cnt = 0
        for srcid, pred_tagsets in pred.items():
            pred_point_tagset = select_point_tagset(pred_tagsets, srcid)
            good_point_tagset = self.zodiac_good_preds.get(srcid, None)
            if not good_point_tagset:
                continue
            if not self.is_same_tagset(pred_point_tagset, good_point_tagset):
                pred_tagsets = [tagset for tagset in pred_tagsets if not is_point_tagset(tagset)]
                pred_tagsets.append(good_point_tagset)
                print('FIXED {0}, {1} -> {2}'.format(srcid,
                                                     pred_point_tagset,
                                                     good_point_tagset))
                fixed_cnt += 1
                pred[srcid] = pred_tagsets
        print('TOTAL_FIXED_POINTS: {0}'.format(fixed_cnt))
        return pred

    def select_informative_samples(self, sample_num=10):
        new_srcids = []
        if len(new_srcids) < sample_num:
            new_srcids += self.scrabble.select_informative_samples(
                sample_num - len(new_srcids))
        return new_srcids
