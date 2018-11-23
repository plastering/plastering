#from plastering.inferencers.quiver import DummyQuiver
from plastering.inferencers.zodiac_new import ZodiacInterface
from plastering.inferencers.scrabble_new import ScrabbleInterface
#from plastering.inferencers.zodiac_interface import ZodiacInterface
from plastering.metadata_interface import *
from plastering.workflow import *
from plastering.helper.common import *
import pdb

# construct a framework dict for referneces.
f_class_dict = {
    'zodiac': ZodiacInterface,
    'quiver': DummyQuiver
}

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
quiver_config = deepcopy(base_config)
building_ttl = 'groundtruth/{0}_brick.ttl'.format(target_building)
quiver_config['config'] = {
    'ground_truth_ttl': building_ttl
}

f_graph = {
    'zodiac': (zodiac_config, {
        'scrabble': (scrabble_config, {
        })
    })
}

# init workflow
workflow = Workflow(target_srcids, f_class_dict, f_graph)
# random srcids to update
new_srcids = random.sample(target_srcids, 10)
for i in range(0, 20):
    workflow.update_model(new_srcids)
    workflow.predict(new_srcids)
    new_srcids = workflow.select_informative_samples(5)
