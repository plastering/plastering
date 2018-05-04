from prompt_toolkit import prompt
from prompt_toolkit.styles import style_from_dict
from prompt_toolkit.token import Token



from ..metadata_interface import *

class ReplUi(object):

    def __init__(self):
        pass

    def display_target(self, srcid):
        try:
            print_rawmetadata(srcid)
        except:
            print('Srcid {0} not found in our DB'.format(srcid)) # or logging
            return None

    def _get_bottom_toolbar_tokens(cli):
        return [(Token.Toolbar, ' This is a toolbar. ')]

    def normalize_tagset(self, raw_tagset):
        tagset = '_'.join(raw_tagset.split()) # TODO: Capitalize if necessary.
        return tagset

    def get_answer(self, srcid, example_type):
        if example_type == 'point_tagset':
            point_tagset = prompt('Point TagSet: ')
            point_tagset = self.normalize_tagset(point_tagset)
            return point_tagset
        elif example_type == 'fullparsing':
            print('Instruction:')
            text = prompt('> ', get_bottom_toolbar_tokens=self._get_bottom_toolbar_tokens)
            # TODO: implement this!


    def ask_example(self, srcid, example_types=[]):
        self.display_target(srcid)
        answers = {}
        for example_type in example_types:
            answer = self.get_answer(srcid, example_type)
            answers[example_type] = answer
        self.store_example(srcid, answers)

    def store_example(self, srcid, answers):
        insert_groundtruth(srcid, **answers)
