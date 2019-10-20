from plastering.inferencers.scrabble_interface import ScrabbleInterface
from plastering.metadata_interface import *
from plastering.uis.cmdline_ui import ReplUi
import pdb

import logging

target_building = 'sdh'
config = {
    'brick_version': '1.0.3',
    'brick_file': 'https://brickschema.org/schema/1.0.3/Brick.ttl',
    'brickframe_file': 'https://brickschema.org/schema/1.0.3/BrickFrame.ttl',
    'crfimpl': 'crfsuite',
    'ir2tagsets.epochs': 1000,
}

# Select labeled srcids (Not all the data are labeled yet.)
labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

scrabble = ScrabbleInterface(target_building=target_building,
                             target_srcids=target_srcids,
                             config=config,
                             logging_configfile='config/logging.yaml',
                             metadata_types=['VendorGivenName'],
                             )
all_tagsets = [tagset.lower().split('#')[-1] for tagset in scrabble.schema_g.get_all_tagsets()]
scrabble.ui = ReplUi(all_tagsets, scrabble.pgid)
#scrabble.learn_auto()
for i in range(0, 20):
    selected_samples = scrabble.select_informative_samples(10)
    scrabble.update_model(selected_samples)
res = scrabble.predict(target_srcids)
res.g.serialize('scrabble_output.ttl', format='turtle')
