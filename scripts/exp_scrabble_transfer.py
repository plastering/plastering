import sys, os
import pdb
import json
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/..')

from plastering.inferencers.scrabble_new import ScrabbleInterface
from plastering.metadata_interface import *
import pdb

EXP_NUM = 2

target_building = sys.argv[1]
source_building = sys.argv[2]
target_buildings = [target_building]
source_buildings = [source_building, target_building]
#target_buildings = ['ebu3b']
#source_buildings = ['ap_m', 'ebu3b']
#target_buildings = ['sdh']
#source_buildings = ['sdh']
sample_num_list = [200, 10]

inferencers = {
    'scrabble': ScrabbleInterface,
}

configs = {
    'scrabble': {
        'config': {
            'use_known_tags': True,
            'n_jobs': 12,
            'tagset_classifier_type': 'MLP',
            'use_brick_flag': False,
            'crfqs': 'confidence',
            'entqs': 'phrase_util',
            'negative_flag': True,
            'sample_num_list': sample_num_list,
        }
    }
}

print('------------------------------')
print('CONFIGS:')
print(configs)
print('------------------------------')


for inferencer_name, Inferencer in inferencers.items():
    for exp_id in range(0, EXP_NUM):
        for target_building in target_buildings:
            # Select labeled srcids (Not all the data are labeled yet.)
            labeled_list = LabeledMetadata.objects(building=target_building)
            target_srcids = [labeled['srcid'] for labeled in labeled_list]
            config = configs[inferencer_name]
            inferencer = Inferencer(target_building =target_building,
                                    target_srcids=target_srcids,
                                    source_buildings=source_buildings,
                                    **config)
            inferencer.learn_auto()
            history = [{
                'metrics': hist['metrics'],
                'learning_srcids': len(hist['total_training_srcids'])
            } for hist in inferencer.history]
            with open('result/pointonly_transfer_{0}_{1}_{2}_{3}.json'
                      .format(inferencer_name, target_building,
                              source_buildings[0], exp_id), 'w') as fp:
                json.dump(history, fp)
