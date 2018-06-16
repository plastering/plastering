import sys, os
import pdb
import json
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/..')

from plastering.inferencers.scrabble_new import ScrabbleInterface
from plastering.metadata_interface import *
import pdb

EXP_NUM = 4

#target_buildings = ['ghc']
#source_buildings = ['ghc']
#target_buildings = ['ebu3b']
#source_buildings = ['ebu3b']
#target_buildings = ['sdh']
#source_buildings = ['sdh']
target_building = sys.argv[1]
target_buildings = [target_building]
source_buildings = [target_building]
sample_num_list = [10]

inferencers = {
    'scrabble': ScrabbleInterface,
}

if int(sys.argv[2]) == 1:
    use_brick_flag = True
elif int(sys.argv[2]) == 0:
    use_brick_flag = False
else:
    raise Exception('incorrect argument')

configs = {
    'scrabble': {
        'config': {
            'use_known_tags': True,
            'n_jobs': 3,
            'tagset_classifier_type': 'MLP',
            'use_brick_flag': use_brick_flag,
            'crfqs': 'confidence',
            'entqs': 'phrase_util',
            'negative_flag': True,
            'sample_num_list': sample_num_list,
        }
    }
}

if use_brick_flag:
    brick_postfix = 'brick'
else:
    brick_postfix = 'nobrick'

print('CONFIG:')
print(configs)
print('======================')


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
            with open('result/pointonly_notransfer_{0}_{1}_{2}_{3}.json'
                      .format(inferencer_name, target_building,
                          exp_id, brick_postfix), 'w') as fp:
                json.dump(history, fp)
