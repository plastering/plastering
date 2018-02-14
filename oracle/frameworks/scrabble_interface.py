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
    def __init__(self, target_building, exp_id='none', conf={
            'source_buildings': ['ebu3b'],
            'source_samples_list': [5],
            'logger_postfix': 'temp',
            'seed_num': 5}):
        super(ScrabbleInterface, self).__init__(conf, exp_id, 'scrabble')
        self.target_building = target_building
        self.source_buildings = conf['source_buildings']
        self.sample_num_list = conf['source_samples_list']
        self.seed_num = conf['seed_num']
        if self.target_building not in self.source_buildings:
            self.source_buildings = self.source_buildings + [self.target_building]
            self.sample_num_list = self.sample_num_list + [self.seed_num]
        conf['use_cluster_flag'] = True
        conf['use_brick_flag'] = True
        conf['negative_flag'] = True
        self.logger_postfix = conf['logger_postfix']

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
                            self.source_buildings,
                            self.target_building,
                            self.sample_num_list,
                            self.building_sentence_dict,
                            self.building_label_dict,
                            self.building_tagsets_dict,
                            conf)
    @exec_measurement
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
        pdb.set_trace()
                                          
    
    @exec_measurement
    def learn_auto2(self, iter_num=1):
        num_sensors_in_gray = 10000
        while num_sensors_in_gray > 0:
            new_srcids = self.zodiac.select_informative_samples_only(10)
            self.update_model(new_srcids)
            num_sensors_in_gray = self.zodiac.get_num_sensors_in_gray()
            pred_point_tagsets = self.zodiac.predict(self.target_srcids)
            for i, srcid in enumerate(self.target_srcids):
                self.pred['tagsets'][srcid] = set([pred_point_tagsets[i]])
            print(num_sensors_in_gray)
            self.evaluate()
        pdb.set_trace()
