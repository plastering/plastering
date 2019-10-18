import sys
import pdb
import json

import pandas as pd
import numpy as np

from plastering.metadata_interface import LabeledMetadata, RawMetadata, get_or_create
pointPostfixes = ['alarm', 'sensor', 'setpoint', 'command', 'status', 'meter']


def pick_point_tagset(tagsets):
    for tagset in tagsets:
        point_postfix = tagset.lower().split('_')[-1]
        if point_postfix in pointPostfixes:
            return tagset
    return 'point'


building = 'sdh'
TAGSET_FILE = 'groundtruth/{0}_tagsets.json'.format(building)
FULLPARSING_FILE = 'groundtruth/{0}_full_parsing.json'.format(building)

fullparsing_dict = json.load(open(FULLPARSING_FILE, 'r'))
tagsets_dict = json.load(open(TAGSET_FILE, 'r'))
for point_name, fullparsing in fullparsing_dict.items():
    srcid = point_name

    metadata_doc = get_or_create(RawMetadata,
                                 srcid=point_name,
                                 building=building,
                                 )
    metadata_doc.metadata = {
        'VendorGivenName': point_name
    }
    metadata_doc.save()

    labeled_doc = get_or_create(LabeledMetadata,
                                srcid=point_name,
                                building=building,
                                )

    labeled_doc.fullparsing = {
        'VendorGivenName': fullparsing,
    }
    tagsets = tagsets_dict[point_name]
    labeled_doc.tagsets = tagsets
    labeled_doc.point_tagset = pick_point_tagset(tagsets)

    labeled_doc.save()
