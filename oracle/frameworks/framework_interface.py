import os
import pdb
import random

import rdflib # May use faster rdf db instead.
import arrow
from rdflib import *

from ..db import *
from ..brick_parser import pointTagsetList as point_tagsets
from ..common import *
from .. import plotter
from .. import *
from ..error import *
from ..rdflib_wrapper import *
#from ..brick_parser import g as brick_g 

PUBLIC_METHODS = ['learn_auto',
                  'predict_proba',
                  'predict',
                  'select_informative_samples',
                  'update_model'
                  ]

def exec_measurement(func):
    def wrapped(*args, **kwargs):
        begin_time = arrow.get()
        res = func(*args, **kwargs)
        end_time = arrow.get()
        print('Execution Time: {0}'.format(end_time - begin_time))
        return res
    return wrapped


class FrameworkInterface(object):
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
                 config={},
                 ):
        super(FrameworkInterface, self).__init__()
        self.exp_id = random.randrange(0,1000)# an identifier for logging/debugging
        self.source_buildings = source_buildings
        self.config = config # future usage
        self.training_srcids = set() # already known srcids
        self.all_point_tagsets = point_tagsets # all the possible point tagsets 
                                               # defined in Brick.
        self.pred = {  # predicted results
            'tagsets': {},
            'point': {}
            }
        self.pred_g = init_graph()
        self.prior_g = init_graph()
        self.pred_probs = {}
        self.target_building = target_building
        self.target_srcids = target_srcids
        self.history = [] # logging and visualization purpose
        self.required_label_types = ['point', 'fullparsing'] # Future purpose
        self.__name__ = framework_name

    def evaluate_points(self):
        curr_log = {
            'training_srcids': self.training_srcids
        }
        score = 0
        for srcid, pred_tagsets in self.pred['tagsets'].items():
            true_tagsets = LabeledMetadata.objects(srcid=srcid)[0].tagsets
            true_point = sel_point_tagset(true_tagsets)
            pred_point = sel_point_tagset(pred_tagsets)
            if true_point == pred_point:
                score +=1
        curr_log['accuracy'] = score / len(self.pred['point'])
        return curr_log

    def evaluate(self):
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

    # ESSENTIAL
    def learn_auto(self, iter_num=1):
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
    def update_model(self, srcids):
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
        self.training_srcids = self.training_srcids.union(set(srcids))
        if not self.training_srcids:
            raise EmptyTrainingSamples()

        # Get examples from the user if labels do not exist
        for srcid in srcids:
            labeled = LabeledMetadata.objects(srcid=srcid)
            if not labeled:
                # TODO: Add function to receive it from actual user.
                pass

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

    def _add_pred_point_result(self, srcid, pred_point):
        self.pred_g.add(self._make_instance_tuple(srcid, pred_point))

    def _make_instance_tuple(self, srcid, pred_point):
        return (URIRef(BASE + srcid), RDF.type, URIRef(BRICK + pred_point))

    # ESSENTIAL
    def predict_proba(self, target_srcids=None):
        # TODO
        self._validate_target_srcids(target_srcids)

    # ESSENTIAL
    def update_prior(self, pred_g):
        self.prior_g = pred_g

    # ESSENTIAL
    def predict(self, target_srcids=None):
        # TODO
        """
        """
        self._validate_target_srcids(target_srcids)
