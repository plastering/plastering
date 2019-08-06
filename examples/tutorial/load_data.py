import sys
import json
import pdb

import pandas as pd
import numpy as np

from plastering.metadata_interface import *
from plastering.common import *
from plastering.helper import load_uva_building, load_ucb_building
from plastering.helper import extract_raw_ucb_labels
from plastering.rdf_wrapper import get_top_class


building = 'bldg'

# Load Raw Metadata
raw_df = pd.read_csv('rawdata/metadata/{0}_rawmetadata.csv'.format(building))
for i, row in raw_df.iterrows():
    srcid = str(row['SourceIdentifier'])
    point = RawMetadata.objects(srcid=srcid, building=building)\
                       .upsert_one(srcid=srcid, building=building)
    point.metadata = {}
    for k in ['BACnetName', 'BACnetDescription', 'VendorGivenName']:
        point.metadata[k] = row[k]
    point.save()

# Load Labeled Metadata
with open('groundtruth/{0}_labeled_metadata.json'.format(building), 'r') as fp:
    data = json.load(fp)

for srcid, doc in data.items():
    labeled = get_or_create(LabeledMetadata, srcid=srcid, building=building)
    for k, v in doc.items():
        setattr(labeled, k, v)
    try:
        labeled.save()
    except:
        continue
