from collections import defaultdict
import pdb

import pandas as pd

from ..metadata_interface import LabeledMetadata, RawMetadata


def elem2list(elem):
    if isinstance(elem, str):
        return elem.split('_')
    else:
        return []


def csv2json(df, key_idx, value_idx):
    keys = df[key_idx].tolist()
    values = df[value_idx].tolist()
    return {k: elem2list(v) for k, v in zip(keys, values)}


def load_data(target_building,
              source_buildings,
              unit_mapping_file='config/unit_mapping.csv',
              bacnettype_mapping_file='config/bacnettype_mapping.csv',
              bacnettype_flag=False,
              metadata_types=['VendorGivenName'],
              ):
    building_sentence_dict = dict()
    building_label_dict = dict()
    building_tagsets_dict = dict()
    known_tags_dict = defaultdict(list)

    units = csv2json(pd.read_csv(unit_mapping_file), 'unit', 'word')
    units[None] = []
    units[''] = []
    bacnettypes = csv2json(pd.read_csv(bacnettype_mapping_file), 'bacnet_type_str', 'candidates')
    bacnettypes[None] = []
    bacnettypes[''] = []
    for building in source_buildings:
        true_tagsets = {}
        label_dict = {}
        for labeled in LabeledMetadata.objects(building=building):
            srcid = labeled.srcid
            true_tagsets[srcid] = labeled.tagsets
            fullparsing = labeled.fullparsing
            labels = {}
            for metadata_type, pairs in fullparsing.items():
                labels[metadata_type] = [pair[1] for pair in pairs]
            label_dict[srcid] = labels

        building_tagsets_dict[building] = true_tagsets
        building_label_dict[building] = label_dict
        sentence_dict = dict()
        for raw_point in RawMetadata.objects(building=building):
            srcid = raw_point.srcid
            metadata = raw_point['metadata']
            sentences = {}
            for clm in metadata_types:
                if clm not in ['BACnetUnit', 'BACnetTypeStr']:
                    sentences[clm] = [c for c in metadata.get(clm, '').lower()]
            sentence_dict[srcid] = sentences
            bacnet_unit = metadata.get('BACnetUnit')
            if bacnet_unit:
                known_tags_dict[srcid] += units[bacnet_unit]
            if bacnettype_flag:
                known_tags_dict[srcid] += bacnettypes[metadata.get('BACnetTypeStr')]
        building_sentence_dict[building] = sentence_dict
    target_srcids = list(building_label_dict[target_building].keys())
    return building_sentence_dict, target_srcids, building_label_dict,\
        building_tagsets_dict, known_tags_dict
