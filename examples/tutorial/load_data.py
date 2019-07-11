import sys
import json
import pdb

import pandas as pd
import numpy as np

#from Oracle.db import OracleDatabase
from plastering.metadata_interface import *
from plastering.common import *
from plastering.helper import load_uva_building, load_ucb_building
from plastering.helper import extract_raw_ucb_labels
from plastering.rdf_wrapper import get_top_class


building = 'bldg'
raw_df = pd.read_csv('rawdata/metadata/bldg_rawmetadata.csv')
for i, row in raw_df.iterrows():
    srcid = str(row['SourceIdentifier'])
    point = RawMetadata.objects(srcid=srcid, building=building)\
                       .upsert_one(srcid=srcid, building=building)
    point.metadata = {}
    for k in ['BACnetName', 'BACnetDescription', 'VendorGivenName']:
        point.metadata[k] = row[k]
    point.save()
