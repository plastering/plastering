import sys, os
import pdb
import json
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/..')

from plastering.inferencers.scrabble_new import ScrabbleInterface
from plastering.metadata_interface import *

exp_order = [2, 3, 0, 1]
#EXP_NUM = 4

target_building = sys.argv[1]
source_building = sys.argv[2]
if source_building == 'none':
    source_building = None
target_buildings = [target_building]
if source_building:
    source_buildings = [source_building, target_building]
    sample_num_list = [200, 10]
else:
    source_buildings = [target_building]
    sample_num_list = [10]

inferencers = {
    'scrabble': ScrabbleInterface,
}

if int(sys.argv[3]) == 1:
    use_brick_flag = True
elif int(sys.argv[3]) == 0:
    use_brick_flag = False
else:
    raise Exception('incorrect argument')

configs = {
    'scrabble': {
        'config': {
            'use_known_tags': True,
            'n_jobs': 3,
            'tagset_classifier_type': 'MLP',
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

if source_building:
    transfer_flag = 'transfer'
else:
    transfer_flag = 'notransfer'


for inferencer_name, Inferencer in inferencers.items():
    for exp_id in exp_order:
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
            if source_building:
                point_file = 'result/pointonly_{0}_{1}_{2}_{3}_{4}_{5}.json'\
                    .format(transfer_flag, inferencer_name, target_building,
                              source_building, exp_id, brick_postfix)
                entity_file = 'result/allentities_{0}_{1}_{2}_{3}_{4}_{5}.json'\
                    .format(transfer_flag, inferencer_name, target_building,
                              source_building, exp_id, brick_postfix)
            else:
                point_file = 'result/pointonly_{0}_{1}_{2}_{3}_{4}.json'\
                    .format(transfer_flag, inferencer_name, target_building,
                              exp_id, brick_postfix)
                entity_file = 'result/allentities_{0}_{1}_{2}_{3}_{4}.json'\
                    .format(transfer_flag, inferencer_name, target_building,
                              exp_id, brick_postfix)

            with open(point_file, 'w') as fp:
                json.dump(history, fp)

            with open(entity_file, 'w') as fp:
                json.dump(history, fp)
