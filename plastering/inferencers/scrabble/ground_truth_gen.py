import pdb
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(choices=['ap_m','ebu3b', 'bml'], dest='building')
args = parser.parse_args()

import pandas as pd

from brick_parser import equipTagsetList as equip_tagsets, \
                         locationTagsetList as location_tagsets,\
                         pointSubclassDict      as  point_subclass_dict,\
                         equipSubclassDict      as  equip_subclass_dict,\
                         locationSubclassDict   as  location_subclass_dict
subclass_dict = dict()
subclass_dict.update(point_subclass_dict)
subclass_dict.update(equip_subclass_dict)
subclass_dict.update(location_subclass_dict)
subclass_dict['networkadapter'] = list()
subclass_dict['none'] = list()
subclass_dict['unknown'] = list()


building = args.building

sensor_df = pd.read_csv('metadata/{0}_sensor_types_location.csv'\
                        .format(building)).set_index('Unique Identifier')

with open('metadata/{0}_label_dict_justseparate.json'\
            .format(building), 'r') as fp:
    label_dict = json.load(fp)
with open('metadata/{0}_sentence_dict_justseparate.json'\
            .format(building), 'r') as fp:
    sentence_dict = json.load(fp)

nonpoint_tagsets = equip_tagsets + location_tagsets + ['networkadapter']

def find_nonpoint_tagsets(tagset):
    if tagset.split('-')[0] in nonpoint_tagsets:
        return tagset
    else:
        return ''

truth_dict = dict()
for srcid, label_list in label_dict.items():
    sentence = sentence_dict[srcid]
    phrase_list = list()
    truth_list = list()
    sentence_meanings = [(token,label) 
                         for token, label 
                         in zip(sentence, label_list) 
                         if label not in ['none', 'unknown']]
    right_identifier_buffer = ''
    for (token, label) in sentence_meanings:
        if label=='leftidentifier':
#            phrase_list[-1] += ('-' + token)
            continue
        elif label=='rightidentifier':
#            right_identifier_buffer += token
            continue

        phrase_list.append(label)
        if right_identifier_buffer:
            phrase_list[-1] += ('-' + right_identifier_buffer)
    truth_list = [phrase
                  for phrase
                  in phrase_list
                  if find_nonpoint_tagsets(phrase)]
    removing_tagsets = list()
    for tagset in truth_list:
        subclasses = subclass_dict[tagset.split('-')[0]]
        if sum([True if tagset in subclasses else False
                for tagset in truth_list]) > 1:
            removing_tagsets.append(tagset)
    for tagset in removing_tagsets:
        truth_list = list(filter(tagset.__ne__, truth_list))
    try:
        truth_list.append(sensor_df['Schema Label'][srcid].replace(' ', '_'))
    except:
        print(srcid, 'failed')
    truth_dict[srcid] = list(set(truth_list))

    # TODO: add all labels to a dict (except point type info)

with open('metadata/{0}_ground_truth.json'.format(building), 'w') as fp:
    json.dump(truth_dict, fp, indent=2)
