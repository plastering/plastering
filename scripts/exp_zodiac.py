import sys, os
import pdb
import json
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/..')
#sys.path.append(os.path.abspath(os.path.join(dir_path + '/..', 'config')))
#sys.path.append(os.path.abspath(os.path.join('..', 'config')))

from plastering.inferencers.zodiac_new import ZodiacInterface
from plastering.metadata_interface import *
import pdb

EXP_NUM = 4

target_building = sys.argv[1]
try:
    source_buildings = [sys.argv[2]]
except:
    source_buildings = []


for exp_id in range(0, EXP_NUM):
    # Select labeled srcids (Not all the data are labeled yet.)
    labeled_list = LabeledMetadata.objects(building=target_building)
    target_srcids = [labeled['srcid'] for labeled in labeled_list]

    zodiac= ZodiacInterface(target_building =target_building,
                            target_srcids=target_srcids,
                            source_buildings=source_buildings)
    zodiac.learn_auto()
    history = [{
        'metrics': hist['metrics'],
        'learning_srcids': len(hist['total_training_srcids'])
    } for hist in zodiac.history]

    if source_buildings:
        outputfile = 'result/pointonly_transfer_{0}_{1}_{2}_{3}.json'\
            .format('zodiac', target_building, source_buildings[0], exp_id)
    else:
        outputfile = 'result/pointonly_transfer_{0}_{1}_{2}.json'\
            .format('zodiac', target_building, exp_id)

    with open(outputfile, 'w') as fp:
        json.dump(history, fp)
