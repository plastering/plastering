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
from jasonhelper import argparser

UCB_BUILDINGS = ['sdh', 'soda', 'ibm']

def parse_ucsd_rawmetadata(building):
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

def parse_fullparsing(building):
    with open('groundtruth/{0}_full_parsing.json'.format(building), 'r') as fp:
        fullparsings = json.load(fp)
    for srcid, fullparsing in fullparsings.items():
        if building in UCB_BUILDINGS:
            fullparsing = {
                'VendorGivenName': fullparsing
            }
        point = LabeledMetadata.objects(srcid=srcid, building=building)\
                               .upsert_one(srcid=srcid, building=building)
        point.fullparsing = fullparsing
        point.save()
    print('Finished adding full parsing')

# add tagsets
def parse_tagsets(building):
    with open('groundtruth/{0}_tagsets.json'.format(building), 'r') as fp:
        true_tagsets = json.load(fp)
    for srcid, tagsets in true_tagsets.items():
        point = LabeledMetadata.objects(srcid=srcid, building=building)\
                               .upsert_one(srcid=srcid, building=building)
        point.tagsets = tagsets
        point.point_tagset = sel_point_tagset(tagsets)
        point.save()

def remove_invalid_srcids(building):
    with open('config/invalid_srcids.json', 'r') as fp:
        invalid_srcids_dict = json.load(fp)
    if building not in invalid_srcids_dict:
        return None
    invalid_srcids = invalid_srcids_dict[building]

    print('{0} invalid points are removed.'.format(len(invalid_srcids)))
    for srcid in invalid_srcids:
        RawMetadata.objects(srcid=srcid).delete()
        LabeledMetadata.objects(srcid=srcid).delete()


if __name__ == '__main__':
    argparser.add_argument('-b', type=str, dest='building', required=True)
    argparser.add_argument('-top',
                           type='bool',
                           dest='topclass_flag',
                           default=False)
    # add raw metadata
    args = argparser.parse_args()
    building = args.building


    if building == 'uva_cse':
        load_uva_building(building)
        print('UVA CSE Done')
        sys.exit()
    elif building in UCB_BUILDINGS:
        extract_raw_ucb_labels()
        basedir = './groundtruth/'
        filenames = {
            'soda': basedir + 'SODA-GROUND-TRUTH',
            'sdh': basedir + 'SDH-GROUND-TRUTH',
            'ibm': basedir + 'IBM-GROUND-TRUTH',
        }
        load_ucb_building(building, filenames[building])
        #parse_tagsets(building)
        parse_fullparsing(building)
    else:
        parse_ucsd_rawmetadata(building)
        parse_tagsets(building)
        parse_fullparsing(building)
    remove_invalid_srcids(building)

    if args.topclass_flag:
        for obj in LabeledMetadata.objects(building=building):
            point_tagset = obj.point_tagset
            topclass_tagset = get_top_class(point_tagset)
            obj.point_tagset = topclass_tagset
            obj.save()
