import pdb
import random

from oracle.frameworks.scrabble_interface2 import ScrabbleInterface
from oracle.db import *


target_building = 'ap_m'

source_buildings = ['ebu3b']
sample_num_list = [5]

labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]
new_srcids = random.sample(target_srcids, 5)

scrabble = ScrabbleInterface(target_building,
                             target_srcids,
                             source_buildings
                             #sample_num_list
                            )
scrabble.update_model(new_srcids)
pdb.set_trace()
pred = scrabble.predict()
pdb.set_trace()
proba = scrabble.predict_proba()
