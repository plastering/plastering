from plastering.inferencers.zodiac import ZodiacInterface
from plastering.metadata_interface import *
from plastering.uis.cmdline_ui import ReplUi
import pdb

import logging

target_building = 'bldg'
config = {
    'brick_version': '1.0.3',
    'brick_file': '/home/jbkoh/repo/Brick/dist/Brick.ttl',
    'brickframe_file': '/home/jbkoh/repo/Brick/dist/BrickFrame.ttl',
}

# Select labeled srcids (Not all the data are labeled yet.)
labeled_list = RawMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

zodiac = ZodiacInterface(target_building=target_building,
                         target_srcids=target_srcids,
                         config=config,
                         logging_configfile='config/logging.yaml',
                         )
all_tagsets = [tagset.lower().split('#')[-1] for tagset in zodiac.schema_g.get_all_tagsets()]
zodiac.ui = ReplUi(all_tagsets, zodiac.pgid)
#zodiac.update_model([])
#pred = zodiac.predict()
zodiac.learn_auto()
