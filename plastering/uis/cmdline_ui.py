import pdb
import sys
from functools import reduce

import numpy as np
from prompt_toolkit import prompt
from prompt_toolkit.styles import style_from_dict
from prompt_toolkit.token import Token
from tabulate import tabulate


from ..metadata_interface import *
from ..common import *

class ReplUi(object):

    def __init__(self, all_tagsets, pgid=None):
        self._init_brick(all_tagsets)
        self.pgid = pgid

    def _init_brick(self, all_tagsets):
        # TODO: Read below from an external file
        non_brick_tagsets = ['none',
                             'rightidentifier',
                             'leftidentifier',
                             'unknown',
                             'pump_flow_status',
                             'networkadapter',
                             'analog_input_sensor',
                             'analog_output_setpoint',
                             'binary_input_sensor',
                             'binary_output_setpoint',
                             'multistate_input_sensor',
                             'multistate_output_setpoint',
                             ]
        # left identifier: contraints meaning of left tagset
        # right identifier: contraints meaning of right tagset
        #TODO: Create a dict with dummy values to speed up lookup if needed.
        self.all_tagsets = all_tagsets + non_brick_tagsets
        splitter = lambda s: s.split('_')
        adder = lambda x, y: x + y
        self.all_tags = list(set(reduce(adder, map(splitter, self.all_tagsets), [])))

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

    def validate_tagset(self, tagset):
        if tagset.split('-')[0] in self.all_tagsets:
            return True
        else:
            return False

    def validate_label(self, label):
        label = label.split('-')[0] # Removing domain-specific names.
        for tag in label.split('_'):
            if tag not in self.all_tags:
                return False
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

    def commands(self, cmd):
        if cmd == 'debug':
            pdb.set_trace()
            return 'debug'
        else:
            raise Exception('Unknown commands: {0}'.format(cmd))

    def get_input(self, task):
        if task == 'receive_label':
            msg = 'label: '
        elif task == 'end_idx':
            msg = 'end_idx: '
        elif task == 'alltagsets':
            msg = 'all tagsets: '

        inp = prompt(msg)

        if inp and inp[0] == '@':
            return self.commands(inp[1:])
        elif task == 'receive_label':
            return self.parse_label(inp)
        elif task == 'end_idx':
            if not inp:
                return None
            else:
                return int(inp)
        elif task == 'alltagsets':
            found_tagsets = []
            if self.validate_tagset(inp):
                found_tagsets.append(inp)
            else:
                print('incorrect tagset: {0}'.format(inp))
            while True:
                print('current tagsets: {0}'.format(found_tagsets))
                inp = prompt(msg)
                if inp == 'done':
                    break
                elif self.validate_tagset(inp):
                    found_tagsets.append(inp)
                else:
                    print('incorrect tagset: {0}'.format(inp))
                    continue
            return list(set(found_tagsets))

    def parse_label(self, label):
        if label == '':
            return 'O'
        elif label == 'l':
            return 'leftidentifier'
        elif label == 'r':
            return 'rightidentifier'
        else:
            return label


    def get_answer_point_tagset(self, srcid, building):
        point_tagset = prompt('Point TagSet: ')
        point_tagset = self.normalize_tagset(point_tagset)
        return point_tagset

    def get_answer_full_parsing(self, srcid, building):
        print('Instruction:')
        done = False
        labeled_metadata = query_labels(
            pgid=self.pgid,
            srcid=srcid,
            building=building,
        ).upsert_one(
            srcid=srcid,
            building=building,
        )
        fullparsing = labeled_metadata[FULL_PARSING]
        metadatas = RawMetadata.objects(srcid=srcid, building=building)\
            .first().metadata
        for metadata_type, sentence in metadatas.items():
            base_idx = 0
            while base_idx < len(sentence):
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
                self.print_sentence_with_pos(sentence, base_idx)
                # 4. Specify which parts to be labeled
                while True:
                    try:
                        end_idx = self.get_input('end_idx')
                        if end_idx == 'debug':
                            continue
                        elif not end_idx:
                            end_idx = base_idx
                            curr_word = sentence[base_idx:end_idx + 1]
                            label = 'O'
                        else:
                            end_idx = int(end_idx)
                            if end_idx < base_idx:
                                raise Exception('end_idx is to low as {0}'
                                                .format(end_idx))
                            curr_word = sentence[base_idx:end_idx + 1]
                            print('-- Curr word: {0}'.format(curr_word))
                            # 5. Specify what the label is
                            while True:
                                label = self.get_input('receive_label')
                                # 5.1. Validate if the label is right according to Brick.
                                if self.validate_label(label):
                                    break
                                else:
                                    print('Not a valid label: {0}'
                                          .format(label))
                        # 6. update the data set.
                        parsed += self.make_bio_tuples(curr_word, label)
                        fullparsing[metadata_type] = parsed
                        base_idx = end_idx + 1
                        break
                    except KeyboardInterrupt:
                        print('Interrupted')
                        sys.exit(0)
                    except Exception as e:
                        print(e)
                        continue
        return fullparsing

    def get_answer_all_tagsets(self, srcid, building):
        print('=================================')
        print_rawmetadata(srcid, building)
        print_fullparsing(srcid, building)
        received_tagsets = self.get_input('alltagsets')
        return received_tagsets

    def get_answer(self, srcid, building, example_type):
        if example_type == POINT_TAGSET:
            return self.get_answer_point_tagset(srcid, building)
        elif example_type == FULL_PARSING:
            return self.get_answer_full_parsing(srcid, building)
        elif example_type == ALL_TAGSETS:
            return self.get_answer_all_tagsets(srcid, building)
        else:
            raise Exception('UI for {0} is not implemented yet'
                            .format(example_type))
        print('done for {0}'.format(srcid))


    def ask_example(self, srcid, building, example_types=[]):
        self.display_target(srcid, building)
        answers = {}
        for example_type in example_types:
            answer = self.get_answer(srcid, building, example_type)
            if answer: #TODO: Do I really need this condition?
                insert_groundtruth(srcid, building, self.pgid, **{example_type: answer})

    def store_example(self, srcid, building, answers):
        insert_groundtruth(srcid, building, self.pgid, **answers)
