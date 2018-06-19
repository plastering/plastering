import os
import sys
import importlib.util
import pdb

from . import Inferencer
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/scrabble')
from ..metadata_interface import *
from ..rdf_wrapper import *
from ..common import *

POINT_POSTFIXES = ['sensor', 'setpoint', 'alarm', 'command', 'meter']

from scrabble import Scrabble # This may imply incompatible imports.
from scrabble.common import *

class ScrabbleInterface(Inferencer):
    """docstring for ScrabbleInterface"""
    def __init__(self,
                 target_building,
                 target_srcids,
                 source_buildings,
                 config=None
                 ):
        super(ScrabbleInterface, self).__init__(
            target_building=target_building,
            target_srcids=target_srcids,
            source_buildings=source_buildings,
            config=config,
            framework_name='scrabble')

        self.target_label_type = ALL_TAGSETS

        if not config:
            config = {}

        # Prepare config for Scrabble object
        if 'seed_num' in config:
            seed_num = config['seed_num']
        else:
            seed_num = 10

        if 'sample_num_list' in config:
            sample_num_list = config['sample_num_list']
        else:
            sample_num_list = [seed_num] * len(set(source_buildings +
                                            [target_building]))

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

        # TODO: This should be migrated into Plastering
        building_sentence_dict, target_srcids, building_label_dict,\
            building_tagsets_dict, known_tags_dict = load_data(target_building,
                                                               source_buildings)
        self.scrabble = Scrabble(target_building,
                                 target_srcids,
                                 building_label_dict,
                                 building_sentence_dict,
                                 building_tagsets_dict,
                                 source_buildings,
                                 sample_num_list,
                                 known_tags_dict,
                                 config=config,
                                 )
        #self.update_model(self.scrabble.learning_srcids)


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
        super(ScrabbleInterface, self).update_model(new_srcids)
        self.scrabble.update_model(new_srcids)

    def postprocessing_pred(self, pred):
        # Currently only ingest point tagsets.
        pred_g = init_graph(empty=True)
        for srcid, tagsets in pred.items():
            point_tagset = sel_point_tagset(tagsets, srcid)
            point_prob = 1 # temporary
            self._add_pred_point_result(pred_g, srcid,
                                        point_tagset, point_prob)
        return pred_g

    def predict(self, target_srcids=None, all_tagsets=False):
        if not target_srcids:
            target_srcids = self.target_srcids
        pred = self.scrabble.predict(target_srcids)
        self.pred_g = self.postprocessing_pred(pred)
        if all_tagsets:
            return self.pred_g, pred # This should be generalized inside
                                     # postprocessing_pred
        else:
            return self.pred_g

    def predict_proba(self, target_srcids=None):
        return self.scrabble.predict_proba(target_srcids)

    def apply_prior_zodiac(self, sample_num):
        if not self.prior_g:
            return []
        instances = get_instance_tuples(self.prior_g)
        good_preds = {}
        for srcid, point_tagset in instances.items():
            triple = (BASE[srcid], RDF.type, BRICK[point_tagset])
            if self.prior_confidences[triple] > 0.5:
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

    def select_informative_samples(self, sample_num=10):
        # Use prior (e.g., from Zodiac.)
        new_srcids = self.apply_prior_zodiac(sample_num)
        if len(new_srcids) < sample_num:
            new_srcids += self.scrabble.select_informative_samples(
                sample_num - len(new_srcids))
        return self.scrabble.select_informative_samples(sample_num)
