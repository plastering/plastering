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

target_building = 10312
source_buildings = [10312, 10320, 10381, 10596, 10606, 10642]
args, config = parse_args_coequipment() # use parse_args_coequipment or parse_args_colocation depending on the task

# TODO: merge the two methods later

random.seed(args.seed)
np.random.seed(args.seed)

print(args, config)

ri = RelationalInference(target_building=target_building,
                         source_buildings=source_buildings,
                         target_srcids=0,
                         config=config,
                         args=args)
