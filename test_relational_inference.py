import random
import numpy as np
from plastering.inferencers.relational_inference_interface import RelationalInference
from plastering.inferencers.relational_inference.relational_inference_helper import parse_args_colocation, \
    parse_args_coequipment

"""
# example for colocation
target_building = 'Soda'
source_buildings = ['Soda']
args, config = parse_args_colocation()
"""

target_building = 'Soda2'
source_buildings = ['Soda', 'Soda2']
args, config = parse_args_coequipment()
# use parse_args_coequipment or parse_args_colocation depending on the task

random.seed(args.seed)
np.random.seed(args.seed)

# print(args, config)

ri = RelationalInference(target_building=target_building,
                         source_buildings=source_buildings,
                         target_srcids=0,
                         config=config,
                         args=args)


