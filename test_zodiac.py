from plastering.inferencers.zodiac_new import ZodiacInterface
#from plastering.inferencers.zodiac_interface import ZodiacInterface
from plastering.metadata_interface import *
import pdb

#target_building = 'ghc'
target_building = 'uva_cse'

config = {
    'brick_version': '1.0.3',
    'brick_file': '/home/jciazdeploy/repo/Brick/dist/Brick.ttl',
    'brickframe_file': '/home/jciazdeploy/repo/Brick/dist/BrickFrame.ttl',
}

# Select labeled srcids (Not all the data are labeled yet.)
labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

zodiac = ZodiacInterface(target_building=target_building,
                         target_srcids=target_srcids,
                         config=config,
                         )
#zodiac.update_model([])
pred = zodiac.predict()
zodiac.learn_auto()
