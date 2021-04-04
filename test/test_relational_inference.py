import random

from plastering.inferencers.relational_inference import *
from plastering.inferencers.relational_inference_helper import *

target_building = 'Soda'
source_buildings = ['Soda']
args, config = parse_args()
# random.seed(args.seed)
# np.random.seed(args.seed)
# don't know where the seed is used


ri = RelationalInference(target_building=target_building,
                         source_buildings=source_buildings,
                         target_srcids=0,
                         config=config,
                         args=args)
print(args)
