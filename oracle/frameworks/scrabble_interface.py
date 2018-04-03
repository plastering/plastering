import os
import sys
import importlib.util
import pdb

from .framework_interface import FrameworkInterface, exec_measurement
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/scrabble')
from ..db import *

POINT_POSTFIXES = ['sensor', 'setpoint', 'alarm', 'command', 'meter']

from scrabble import Scrabble # This may imply incompatible imports.


class ScrabbleInterface(FrameworkInterface):
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

        if not config:
            config = {}

        # Prepare config for Scrabble object
        if 'sample_num_list' in config:
            sample_num_list = config['sample_num_list']
        else:
            sample_num_list = [0] * (len(source_buildings) + 1) # +1 for target
        """
            for building in self.source_buildings:
                sample_cnt = 0
                labeled_list = LabeledMetadata.objects(building=target_building)
                for labeled in labeled_list:
                    if labeled.fullparsing:
                        sample_cnt += 1
                sample_num_list.append(sample_cnt)
        if 'seed_num' not in config:
            seed_num = 10
        else:
            seed_num = config['seed_num']
        """

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

        column_names = ['VendorGivenName',
                         'BACnetName',
                         'BACnetDescription']

        self.building_sentence_dict = dict()
        self.building_label_dict = dict()
        self.building_tagsets_dict = dict()
        for building in self.source_buildings:
            true_tagsets = {}
            label_dict = {}
            for labeled in LabeledMetadata.objects(building=building):
                srcid = labeled.srcid
                true_tagsets[srcid] = labeled.tagsets
                fullparsing = None
                for clm in column_names:
                    one_fullparsing = [i[1] for i in labeled.fullparsing[clm]]
                    if not fullparsing:
                        fullparsing = one_fullparsing
                    else:
                        fullparsing += ['O'] + one_fullparsing
                        #  This format is alinged with the sentence 
                        #  conformation rule.
                label_dict[srcid] = fullparsing

            self.building_tagsets_dict[building] = true_tagsets
            self.building_label_dict[building] = label_dict
            sentence_dict = dict()
            for raw_point in RawMetadata.objects(building=building):
                srcid = raw_point.srcid
                if srcid in true_tagsets:
                    metadata = raw_point['metadata']
                    sentence = None
                    for clm in column_names:
                        if not sentence:
                            sentence = [c for c in metadata[clm].lower()]
                        else:
                            sentence += ['\n'] + \
                                        [c for c in metadata[clm].lower()]
                    sentence_dict[srcid]  = sentence
            self.building_sentence_dict[building] = sentence_dict

        # Validation of the dataset
        for building in self.source_buildings:
            for srcid, label_pairs in self.building_label_dict[building]\
                                          .items():
                assert len(label_pairs) == \
                           len(self.building_sentence_dict[building][srcid])

        self.scrabble = Scrabble(
            target_building=self.target_building,
            target_srcids=self.target_srcids,
            building_label_dict=self.building_label_dict,
            building_sentence_dict=self.building_sentence_dict,
            building_tagsets_dict=self.building_tagsets_dict,
            source_buildings=self.source_buildings,
            source_sample_num_list=sample_num_list,
            conf=config,
            learning_srcids=[])

    def learn_auto(self, iter_num=1):
        params = (self.source_buildings,
                  self.sample_num_list,
                  self.target_building)
        self.learned_srcids = []
        params = {
            'use_cluster_flag': True,
            'use_brick_flag': True,
            'negative_flag': True,
            'target_building': self.target_building,
            'building_list': self.source_buildings,
            'sample_num_list': self.scrabble.sample_num_list
            }
        #self.scrabble.char2tagset_iteration(iter_num, self.logger_postfix, *params)
        step_data = {'iter_num':0,
                     'next_learning_srcids': self.scrabble.get_random_srcids(
                                            self.scrabble.building_srcid_dict,
                                            self.source_buildings,
                                            self.sample_num_list),
                     'model_uuid': None}
        step_datas = [step_data]
        step_datas.append(self.scrabble.char2tagset_onestep(step_data,
                                                            **params))

    def update_model(self, srcids):
        self.scrabble.update_model(srcids)

    def predict(self, target_srcids=None):
        return self.scrabble.predict(target_srcids)

    def predict_proba(self, target_srcids=None):
        return self.scrabble.predict_proba(target_srcids)

    def select_informative_samples(self, sample_num=10):
        return self.scrabble.select_informative_samples_only(sample_num)
