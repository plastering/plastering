import json
import pdb

import pandas as pd
import numpy as np

#from Oracle.db import OracleDatabase
from oracle.db import *
from oracle.common import *
from jasonhelper import argparser

argparser.add_argument('-b', type=str, dest='building', required=True)

#db = OracleDatabase()

# add raw metadata
args = argparser.parse_args()
building = args.building
rawdf = pd.read_csv('rawdata/metadata/{0}_rawmetadata.csv'\
                        .format(building), index_col='SourceIdentifier')
for srcid, row in rawdf.iterrows():
    point = RawMetadata.objects(srcid=srcid, building=building)\
                       .upsert_one(srcid=srcid, building=building)
    for k, v in row.items():
        if not isinstance(v, str):
            if np.isnan(v):
                v = ''
        point.metadata[k] = v
    point.save()

print('Finished adding raw metadata')

# add labeled metadata
with open('groundtruth/{0}_full_parsing.json'.format(building), 'r') as fp:
    fullparsings = json.load(fp)
for srcid, fullparsing in fullparsings.items():
    point = LabeledMetadata.objects(srcid=srcid, building=building)\
                           .upsert_one(srcid=srcid, building=building)
    point.fullparsing = fullparsing
    point.save()
print('Finished adding full parsing')

# add tagsets
with open('groundtruth/{0}_tagsets.json'.format(building), 'r') as fp:
    true_tagsets = json.load(fp)
for srcid, tagsets in true_tagsets.items():
    point = LabeledMetadata.objects(srcid=srcid, building=building)\
                           .upsert_one(srcid=srcid, building=building)
    point.tagsets = tagsets
    point.point_tagset = sel_point_tagset(tagsets)
    point.save()

print('Finished adding tagsets')
