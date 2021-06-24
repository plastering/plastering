import random
import numpy as np
# from plastering.inferencers.relational_inference_interface import RelationalInference
from plastering.inferencers.relational_inference.relational_inference_helper import parse_args_colocation, parse_args_coequipment

target_building = 'Soda'
source_buildings = ['Soda']
args, config = parse_args_colocation()

random.seed(args.seed)
np.random.seed(args.seed)

print(args, config)

# ri = RelationalInference(target_building=target_building,
#                          source_buildings=source_buildings,
#                          target_srcids=0,
#                          config=config,
#                          args=args)
