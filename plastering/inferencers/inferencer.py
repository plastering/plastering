import os
import time
import pdb
import random
from copy import deepcopy

import arrow

from ..metadata_interface import *
from ..common import *
from .. import plotter
from .. import *
from ..error import *
from ..rdf_wrapper import *
from ..evaluator import *
from ..uis import *

PUBLIC_METHODS = ['learn_auto',
                  'predict_proba',
                  'predict',
                  'select_informative_samples',
                  'update_model'
                  ]

class Inferencer(object):
    """
    # input parameters
    - target_building (str): name of the target building. this can be arbitrary later
    - source_buildings (list(str)): list of buildings already known.
    - conf: dictionary of other configuration parameters.
    """

    def __init__(self,
                 target_building,
                 target_srcids,
                 source_buildings=[],
                 source_sample_num_list=[],
                 framework_name='dummy_framework',
                 ui=None,
                 required_label_types=[POINT_TAGSET, FULL_PARSING, ALL_TAGSETS],
                 target_label_type=POINT_TAGSET,
                 pgid=None,
                 config={},
                 ):
        super(Inferencer, self).__init__()
        self.target_label_type = target_label_type
        self.exp_id = random.randrange(0,1000)# an identifier for logging/debugging
        self.source_buildings = source_buildings
        if 'brick_version' in config:
            self.brick_version = config['brick_version']
        else:
            self.brick_version = '1.0.2'
        if 'brick_file' in config:
            self.brick_file = config['brick_file']
        else:
            self.brick_file = 'brick/Brick_1_0_2.ttl'
        if 'brickframe_file' in config:
            self.brickframe_file = config['brickframe_file']
        else:
            self.brickframe_file = 'brick/BrickFrame_1_0_2.ttl'
        if 'triplestore_type' in config:
            self.triplestore_type = config['triplestore_type']
        else:
            self.triplestore_type = RDFLIB
        if 'hotstart' in config:
            self.hotstart = config['hotstart']
        else:
            self.hotstart = False
        self.pgid = pgid
        self.config = config
        self.training_srcids = [] # already known srcids
        self.pred = {  # predicted results
            'tagsets': {},
            'point': {}
            }
        self.template_g = self.new_graph(empty=True)
        self.prior_g = self.new_graph(empty=True)
        self.prior_confidences = {}
        self.schema_g = self.new_graph(empty=False)
        self.pred_probs = {}
        self.target_building = target_building
        self.target_srcids = target_srcids
        self.history = [] # logging and visualization purpose
        self.required_label_types = required_label_types
        if ui:
            self.ui = ui
        else:
            schema_g = self.new_graph(empty=False)
            self.ui = ReplUi(schema_g, self.pgid)
        self.__name__ = framework_name + '-' + str(self.exp_id)
        self.result_filename = './result/{0}_history.json'\
            .format(self.__name__)
        self.pred_g = self.new_graph(empty=True)
        self.pred_confidences = {}

    def query_labels(self, **query):
        return query_labels(self.pgid, **query)

    def evaluate_points_dep(self, pred):
        curr_log = {
            'training_srcids': self.training_srcids
        }
        score = 0
        for srcid, pred_tagsets in pred['tagsets'].items():
            true_tagsets = self.query_labels(srcid=srcid)[0].tagsets
            true_point = sel_point_tagset(true_tagsets)
            pred_point = sel_point_tagset(pred_tagsets)
            if true_point == pred_point:
                score += 1
        curr_log['accuracy'] = score / len(pred['point'])
        return curr_log

    def evaluate_dep(self, pred):
        points_log = self.evaluate_points()
        log = {
            'points': points_log
        }
        self.history.append(log)

    def plot_result_point(self):
        srcid_nums = [len(log['points']['learned_srcids']) for log in self.history]
        accs = [log['points']['accuracy'] for log in self.history]
        fig, _ = plotter.plot_multiple_2dline(srcid_nums, [accs])
        for ax in fig.axes:
            ax.set_grid(True)
        plot_name = '{0}_points_{1}.pdf'.format(self.framework_name, self.exp_id)
        plotter.save_fig(fig, plot_name)

    def _ask_example_pointonly(self, srcid):
        point_tagset = input('Its point tagset: ')
        insert_groundtruth(srcid, point_tagset=point_tagset)

    def ask_example(self, srcid, required_label_types=[]):
        if not required_label_types:
            required_label_types = self.required_label_types
        self.ui.ask_example(srcid, self.target_building, required_label_types)

    # ESSENTIAL
    def learn_auto(self, iter_num=1, inc_num=1):
        """Learn from the scratch to the end.

        This executes the learning mechanism from the ground truth.
        It iterates for the given amount of the number.
        Basic procedure is iterating this:
            ```python
            f = Framework()
            while not final_condition:
                new_srcids = f.select_informative_samples(10)
                f.update_model(new_srcids)
                self.pred['tagsets'] = XXX
                f.evaluate()
                final_condition = f.get_final_condition()
            ```

        Args:
            iter_num (int): total iteration number.

        Returns:
            None

        Byproduct:

        """
        pass

    # ESSENTIAL
    def update_model(self, new_srcids):
        """Update model with given newly added srcids.

        This update the model based on the newly added srcids.
        Relevant data for the srcids are given from the ground truth data.
        We can later add an interactive function for users to manually add them

        Args:
            srcids (list(str)): The model will be updated based on the given
                                srcids.
        Returns:
            None

        Byproduct:
            The model will be updated, which can be used for predictions.
        """
        for srcid in new_srcids:
            if srcid in self.training_srcids:
                print('WARNING: {0} already exists in training set, not adding'
                      .format(srcid))
            else:
                self.training_srcids.append(srcid)
        if not self.training_srcids:
            print('WARNING: New srcids are not given')

        # Get examples from the user if labels do not exist
        for srcid in new_srcids:
            labeled = self.query_labels(srcid=srcid).first()
            if not labeled:
                self.ask_example(srcid)
            else:
                for label_type in self.required_label_types:
                    if not labeled[label_type]:
                        self.ask_example(srcid, [label_type])

    # ESSENTIAL
    def select_informative_samples(self, sample_num):
        """Select the most informative N samples from the unlabeled data.

        This function is mainly used by active function frameworks to select
        the most informative samples to ask to the domain experts.
        The chosen samples are again fed back to the model updating mechanisms.

        Args:
            sample_num (int): The number of samples to be chosen

        Returns:
            new_srcids (list(str)): The list of srcids.

        Byproducts:
            None
        """
        pass


    def _validate_target_srcids(self, srcids):
        if not srcids:
            srcids = self.target_srcids

        # Check if the srcid's data are all provided.
        # Assuming self.target_srcids are given.
        for srcid in srcids:
            if srcid not in self.target_srcids:
                raise Exception('The raw data of {0} not given yet'
                                    .format(srcid))

    # ESSENTIAL
    def predict_proba(self, target_srcids=None):
        # TODO
        self._validate_target_srcids(target_srcids)

    # ESSENTIAL
    def update_prior(self, pred_g, pred_confidences={}):
        self.prior_g = pred_g
        self.prior_confidences = pred_confidences

    # ESSENTIAL
    def predict(self, target_srcids=None):
        # TODO
        """
        """
        self._validate_target_srcids(target_srcids)

    def _get_true_labels(self, srcids, label_type):
        """
        Input:
          - target_srcids
          - label_type: one of POINT_TAGSET, FULL_PARSING defined in common.py
        """
        truths = {}
        for srcid in srcids:
            objs = self.query_labels(srcid=srcid)
            if not objs:
                raise Exception('No {0} labels found for {1}'
                                .format(label_type, srcid))
            truths[srcid] = objs.first()[label_type]
        return truths

    def evaluate(self, target_srcids):
        """
        Input:
          - target_srcids
          - label_type: one of POINT_TAGSET, FULL_PARSING defined in common.py
        """
        pred_g = self.predict(target_srcids)
        metrics = {}

        if self.target_label_type in [POINT_TAGSET, ALL_TAGSETS]:
            truth = self._get_true_labels(target_srcids, POINT_TAGSET)
            pred = pred_g.get_instance_tuples()
            metrics['f1'] = get_multiclass_micro_f1(truth, pred)
            metrics['macrof1'] = get_multiclass_macro_f1(truth, pred)
            curr_pred = pred

        if self.target_label_type in [ALL_TAGSETS]:
            pred_g, pred = self.predict(target_srcids, True)
            pred = {srcid: list(pred_tagsets)
                    for srcid, pred_tagsets in pred.items()}
            truth = self._get_true_labels(target_srcids, ALL_TAGSETS)
            metrics['f1-all'] = get_micro_f1(truth, pred)
            metrics['macrof1-all'] = get_macro_f1(truth, pred)
            metrics['accuracy'] = get_accuracy(truth, pred)
            curr_pred = pred

        target_building_training_srcids = \
            [srcid for srcid in self.training_srcids
             if RawMetadata.objects(srcid=srcid,
                                    building=self.target_building).count()]
        total_training_srcids = deepcopy(self.training_srcids)
        curr_eval = {
            'metrics': metrics,
            'total_training_srcids': total_training_srcids,
            'target_building_training_srcids': target_building_training_srcids,
            'pred': curr_pred
        }
        self.history.append(curr_eval)
        return curr_eval

    def filter_prior(self, min_prob):
        for triple, prob in self.prior_confidences.items():
            if prob < min_prob:
                self.prior_g.remove(triple)


    def new_graph(self, empty=True):
        return BrickGraph(empty,
                          version=self.brick_version,
                          brick_file=self.brick_file,
                          brickframe_file=self.brickframe_file,
                          triplestore_type=self.triplestore_type,
                          )
    def add_pred(self, pred_g, pred_confidences,
                 srcid, pred_point, pred_prob):
        triple = pred_g.add_pred_point_result(srcid, pred_point)
        if triple is None:
            triple = srcid
        pred_confidences[triple] = pred_prob


