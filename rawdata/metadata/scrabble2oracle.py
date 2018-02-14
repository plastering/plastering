import json
import pdb
from functools import reduce
from copy import deepcopy

import pandas as pd
import numpy as np

from jasonhelper import argparser

# Helper functions

adder = lambda x, y: x + y

def normalize_phrase(phrase):
    if type(phrase) == str:
        return phrase.lower()
    else:
        if np.isnan(phrase):
            return ''
        else:
            assert False

argparser.add_argument('-b', 
                       type=str, 
                       help='Target building name', 
                       dest='building')
args = argparser.parse_args()

building = args.building


# Transform parsed lables used in Scrabble for Oracle.

df = pd.read_csv('{0}_rawmetadata.csv'.format(building))\
           .set_index('SourceIdentifier')
with open('{0}_char_label_dict.json'.format(building), 'r') as fp:
    char_label_dict = json.load(fp)

parse_dict = dict()

used_col_names = ['BACnetName', 'VendorGivenName', 'BACnetDescription']
for srcid, char_label_pairs in char_label_dict.items():
    row = df.loc[srcid]
    sentence = reduce(adder, [normalize_phrase(row[col_name]) + '\n' 
                              for col_name in used_col_names], '')
    try:
        assert ''.join([pair[0] for pair in char_label_pairs]) == sentence
    except:
        pdb.set_trace()
    sent_it = iter(char_label_pairs)
    pairs = deepcopy(char_label_pairs)
    col_parse_dict = dict()
    for col_name in used_col_names:
        phrase = normalize_phrase(row[col_name])
        col_parse_dict[col_name] = list()
        for _ in phrase:
            char_label_pair = char_label_pairs.pop(0)
            col_parse_dict[col_name].append(char_label_pair)
        [char, _] = char_label_pairs.pop(0)
        assert char == '\n'
    parse_dict[srcid] = dict(col_parse_dict)
    assert len(char_label_pairs) == 0

with open('../../groundtruth/{0}_full_parsing.json'.format(building), 'w') \
         as fp:
    json.dump(parse_dict, fp, indent=2)

