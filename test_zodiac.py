from plastering.inferencers.zodiac_new import ZodiacInterface
#from plastering.inferencers.zodiac_interface import ZodiacInterface
from plastering.metadata_interface import *
import pdb

target_building = 'ebu3b'
#target_building = 'uva_cse'

# Select labeled srcids (Not all the data are labeled yet.)
labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

zodiac = ZodiacInterface(target_building=target_building,
                         target_srcids=target_srcids)
#zodiac.update_model([])
pred = zodiac.predict()
zodiac.learn_auto()
pred = zodiac.predict()
proba = zodiac.predict_proba()
