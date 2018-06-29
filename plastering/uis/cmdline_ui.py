import pdb

import numpy as np

from prompt_toolkit import prompt
from prompt_toolkit.styles import style_from_dict
from prompt_toolkit.token import Token
from tabulate import tabulate


from ..metadata_interface import *
from ..common import *

class ReplUi(object):

    def __init__(self):
        pass

    def display_target(self, srcid, building):
        if not RawMetadata.objects(srcid=srcid, building=building):
            raise Exception('Srcid {0} not found in our DB'.format(srcid))
        print_rawmetadata(srcid, building)

    def _get_bottom_toolbar_tokens(cli):
        return [(Token.Toolbar, ' This is a toolbar. ')]

    def normalize_tagset(self, raw_tagset):
        tagset = '_'.join(raw_tagset.split()) # TODO: Capitalize if necessary.
        return tagset

    def print_sentence_with_pos(self, sentence, base=0):
        num_levels = int(np.log10(len(sentence))) + 1
        for level in reversed(list(range(0, num_levels))):
            divider = np.power(10, level)
            line = ''
            for i, c in enumerate(sentence[base:]):
                istr = str(i + base)
                if len(istr) <= level :
                    curr_digit = '0'
                else:
                    curr_digit = istr[len(istr) - level - 1]
                if curr_digit == '0' and level > 0:
                    line += ' '
                else:
                    line += curr_digit
            print(line)
        print(sentence[base:])

    def validate_label(self, label):
        #TODO Implement this
        return True

    def make_bio_tuples(self, word, label):
        tup = []
        if label == 'O':
            return [[c, 'O'] for c in word]
        else:
            tup.append([word[0], 'B_' + label])
            for c in word[1:]:
                tup.append([c, 'I_' + label])
            return tup

    def receive_label(self):
        label = prompt('label: ')
        if label == '':
            return 'O'
        elif label == 'l':
            return 'leftidentifier'
        elif label == 'r':
            return 'rightidentifier'
        else:
            return label


    def get_answer(self, srcid, building, example_type):
        if example_type == POINT_TAGSET:
            point_tagset = prompt('Point TagSet: ')
            point_tagset = self.normalize_tagset(point_tagset)
            return point_tagset
        elif example_type == FULL_PARSING:
            base = 0
            print('Instruction:')
            done = False
            labeled_metadata = LabeledMetadata.objects(srcid=srcid,
                                                       building=building)\
                .upsert_one(srcid=srcid, building=building)
            fullparsing = labeled_metadata[FULL_PARSING]
            metadatas = RawMetadata.objects(srcid=srcid, building=building)\
                .first().metadata
            for metadata_type, sentence in metadatas.items():
                while base < len(sentence):
                    print('=================================')
                    # 1. Print the entire raw metadata
                    print_rawmetadata(srcid, building)
                    # 2. Print the labeled data so far.
                    print('***************Labeled******************')
                    parsed = fullparsing.get(metadata_type, [])
                    print('Metadata Type: {0}'.format(metadata_type))
                    labeled_df = pd.DataFrame({
                        'words': [row[0] for row in parsed],
                        'labels': [row[1] for row in parsed]
                    })
                    print(tabulate(labeled_df, headers='keys', tablefmt='psql'))

                    # 3. Print the unlabeled data so far.
                    print('***************Unlabeled******************')
                    print('Metadata Type: {0}'.format(metadata_type))
                    self.print_sentence_with_pos(sentence, base)
                    # 4. Specify which parts to be labeled
                    while True:
                        try:
                            end_idx = prompt('end idx: ')
                            if not end_idx:
                                end_idx = base
                                curr_word = sentence[base:end_idx + 1]
                                label = 'O'
                            else:
                                end_idx = int(end_idx)
                                if end_idx < base:
                                    raise Exception('end_idx is to low as {0}'
                                                    .format(end_idx))
                                curr_word = sentence[base:end_idx + 1]
                                print('-- Curr word: {0}'.format(curr_word))
                                # 5. Specify what the label is
                                while True:
                                    label = self.receive_label()
                                    # 5.1. Validate if the label is right according to Brick.
                                    if self.validate_label(label):
                                        break
                            # 6. update the data set.
                            parsed += self.make_bio_tuples(curr_word, label)
                            fullparsing[metadata_type] = parsed
                            base = end_idx + 1
                            break
                        except Exception as e:
                            print(e)
                            continue
            labeled_metadata[FULL_PARSING] = fullparsing
            labeled_metadata.save()
        else:
            raise Exception('UI for {0} is not implemented yet'
                            .format(example_type))
        print('done for {0}'.format(srcid))


    def ask_example(self, srcid, building, example_types=[]):
        self.display_target(srcid, building)
        answers = {}
        for example_type in example_types:
            answer = self.get_answer(srcid, building, example_type)
            if answer:
                answers[example_type] = answer
        self.store_example(srcid, building, answers)

    def store_example(self, srcid, building, answers):
        insert_groundtruth(srcid, building, **answers)
