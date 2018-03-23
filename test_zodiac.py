from oracle.frameworks.zodiac_interface2 import ZodiacInterface
from oracle.db import *

building = 'ap_m'

labeled_list = LabeledMetadata.objects(building=building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

zodiac = ZodiacInterface(building, target_srcids)
zodiac.learn_auto()
