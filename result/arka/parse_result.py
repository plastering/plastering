import os
import pdb
import re

import pandas as pd

target_building = 'sdh'
currfile = __file__
base_dir = os.path.dirname(currfile)
target_dir = base_dir + '/' + target_building
cluster_sizes = {}
for filename in os.listdir(target_dir):
    df = pd.read_csv(target_dir + '/' + filename)
    df.columns = df.columns.str.strip()
    cluster_id = int(re.findall('\\d+', filename)[0])
    coverages = df['fullyQualified'].tolist()
    pdb.set_trace()
