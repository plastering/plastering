import random

from plastering.inferencers.relational_inference import *
from plastering.inferencers.relational_inference_helper import *
import matplotlib.pyplot as plt


target_building = 'Soda'
source_buildings = ['Soda']
args, config = parse_args()
# # random.seed(args.seed)
# # np.random.seed(args.seed)
# # don't know where the seed is used
#
ri = RelationalInference(target_building=target_building,
                         source_buildings=source_buildings,
                         target_srcids=0,
                         config=config,
                         args=args)


# read_colocation_data("Soda")

# data = np.genfromtxt("./rawdata/metadata/Soda/SOD34BLD_C_SAS.csv", delimiter=",", names=["x", "y"])
# plt.plot(data['x'], data['y'])
#
# plt.show()