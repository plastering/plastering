from .db import *

# TODO: This needs to be parameterized
column_names = ['VendorGivenName', 
                 'BACnetName', 
                 'BACnetDescription']


def data_loader(target_building, source_buildings):
    building_sentence_dict = dict()
    building_label_dict = dict()
    building_tagsets_dict = dict()
    for building in source_buildings:
        true_tagsets = {}
        label_dict = {}
        for labeled in LabeledMetadata.objects(building=building):
            srcid = labeled.srcid
            true_tagsets[srcid] = labeled.tagsets
            fullparsing = None
            for clm in column_names:
                one_fullparsing = [i[1] for i in labeled.fullparsing[clm]]
                if not fullparsing:
                    fullparsing = one_fullparsing
                else:
                    fullparsing += ['O'] + one_fullparsing
                    #  This format is alinged with the sentence 
                    #  configormation rule.
            label_dict[srcid] = fullparsing

        building_tagsets_dict[building] = true_tagsets
        building_label_dict[building] = label_dict
        sentence_dict = dict()
        for raw_point in RawMetadata.objects(building=building):
            srcid = raw_point.srcid
            if srcid in true_tagsets:
                metadata = raw_point['metadata']
                sentence = None
                for clm in column_names:
                    if not sentence:
                        sentence = [c for c in metadata[clm].lower()]
                    else:
                        sentence += ['\n'] + \
                                    [c for c in metadata[clm].lower()]
                sentence_dict[srcid]  = sentence
        building_sentence_dict[building] = sentence_dict
    return building_sentence_dict, building_label_dict, building_tagsets_dict
