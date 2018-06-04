import pdb
import json
from functools import reduce

from ..metadata_interface import *
from ..common import *

with open('./groundtruth/ucb_brick_map.json', 'r') as fp:
    brick_map = json.load(fp)

def extract_raw_ucb_labels():
    buildings = ['SODA', 'SDH', 'IBM']
    labels = set()
    example_dict = {}
    for building in buildings:
        filename='./groundtruth/{0}-GROUND-TRUTH'.format(building)
        with open(filename, 'r') as fp:
            rawlines = [line[:-1] for line in fp.readlines()]

        for i, sentence in enumerate(rawlines[::2]):
            i *= 2
            encoded = rawlines[i+1]
            splitted = encoded.split(',')
            for elem in splitted:
                [label, word, t] = elem.split(':')
                if t == 'c':
                    labels.add(label)
                    example_dict[label] = sentence
    with open('groundtruth/ucb_raw_labels.txt', 'w') as fp:
        fp.write('{\n')
        for label in labels:
            fp.write('  "{0}": \n'.format(label))
        fp.write('}')

    with open('groundtruth/ucb_label_sentence_map.json', 'w') as fp:
        json.dump(example_dict, fp, indent=2)

def make_bio_word_label(word, label):
    if not word:
        print('WARNING: {0} is mapped to an empty string'.format(label))
        return []
    pairs = [[word[0], 'B_' + label]]
    pairs += [[c, 'I_' + label] for c in word[1:]]
    return pairs

def load_ucb_building(building='soda',
                      filename='./groundtruth/SODA-GROUND-TRUTH'):
    assert building in ['soda', 'sdh', 'ibm'], \
        'Srong building name: {0}'.format(building)

    with open(filename, 'r') as fp:
        rawlines = [line[:-1] for line in fp.readlines()]
    word_tagsets_dict = {}
    sentence_dict = {}
    tagsets_dict = {}
    tagsets_parsing = {}
    for i, sentence in enumerate(rawlines[::2]):
        i *= 2
        srcid = sentence
        words = []
        labels = []
        types = []
        tagsets = set()
        encoded = rawlines[i+1]
        splitted = encoded.split(',')
        for elem in splitted:
            [label, word, t] = elem.split(':')
            words.append(word)
            labels.append(label)
            types.append(t)
            if t == 'c':
                if label not in brick_map:
                    print('Label not found in Brick map: {0}'.format(label))
                    print('Sentence: {0}'.format(sentence))
                    pdb.set_trace()
                tagset = brick_map[label]
                tagsets.add(tagset.lower())
        tagsets = list(tagsets)
        point_tagset = sel_point_tagset(tagsets)
        tagsets_dict[srcid] = tagsets
        sentence_dict[srcid] = words
        word_tagsets = ['leftidentifier' if label[-3:] == '-id'
                                    else brick_map[label]
                                    for label in labels]
        word_tagsets_dict[srcid] = word_tagsets
        tagsets_parsing = reduce(adder, [make_bio_word_label(word, word_tagset)
                                         for word, word_tagset
                                         in zip(words, word_tagsets)])
        raw_obj = RawMetadata.objects(srcid=srcid)\
            .upsert_one(srcid=srcid, building=building)
        raw_obj.metadata['VendorGivenName'] = srcid
        raw_obj.save()

        labeled_obj = LabeledMetadata.objects(srcid=srcid)\
            .upsert_one(srcid=srcid, building=building)
        labeled_obj.point_tagset = point_tagset
        labeled_obj.tagsets_parsing
        labeled_obj.tagsets = tagsets
        labeled_obj.save()
