from oracle.frameworks.zodiac_interface import ZodiacInterface
from oracle.db import *

target_building = 'ap_m'

# Select labeled srcids (Not all the data are labeled yet.)
labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

zodiac = ZodiacInterface(target_building, target_srcids)
zodiac.learn_auto()
pred = zodiac.predict()
proba = zodiac.predict_proba()
