import pdb
import json

from ..metadata_interface import *

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

def load_ucb_building(building='soda',
                      filename='./groundtruth/SDH-GROUND-TRUTH'):
    assert building in ['soda', 'sdh', 'ibm'], \
        'Srong building name: {0}'.format(building)

    with open(filename, 'r') as fp:
        rawlines = [line[:-1] for line in fp.readlines()]
    word_label_dict = {}
    sentence_dict = {}
    tagsets_dict = {}
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
        tagsets_dict[srcid] = list(tagsets)
        sentence_dict[srcid] = words
        word_label_dict[srcid] = labels
