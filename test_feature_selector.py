from plastering.feature_selector import *
import sys

target_building = sys.argv[1]
load_from_file = int(sys.argv[2])
method = "lsvc"

fs = feature_selector(target_building, method, load_from_file)
fs.run_auto()
