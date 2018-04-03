import pdb
from copy import deepcopy
import random

from oracle.workflow import *
from oracle.db import *
from oracle.frameworks.zodiac_interface import ZodiacInterface
from oracle.frameworks.scrabble_interface import ScrabbleInterface
from oracle.helper import *

# construct a framework dict for referneces.
f_class_dict = {
    'zodiac': ZodiacInterface,
    'scrabble': ScrabbleInterface
}

# Base configurations
target_building = 'ebu3b'
source_buildings = ['ap_m', 'ebu3b']
building_sentence_dict, building_label_dict, building_tagsets_dict = \
    data_loader(target_building, source_buildings)
target_srcids = list(building_tagsets_dict[target_building].keys())
base_config = {
    'target_building': target_building,
    'target_srcids': target_srcids,
    'source_buildings': source_buildings
}
zodiac_config = deepcopy(base_config)
scrabble_config = deepcopy(base_config)
scrabble_config['config'] = {
    'sample_num_list': [5]
}

# Framework Graph
f_graph = {
    'zodiac': (zodiac_config, {
        'scrabble': (scrabble_config, {}),
        'zodiac': (zodiac_config, {})
    })}

# init workflow
workflow = Workflow(target_srcids, f_class_dict, f_graph)
# random srcids to update
new_srcids = random.sample(target_srcids, 20)
workflow.update_model(new_srcids)
workflow.predict(new_srcids)
