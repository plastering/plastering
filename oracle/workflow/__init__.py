import pdb
import random

from ..frameworks.framework_interface import PUBLIC_METHODS
from ..frameworks.framework_interface import FrameworkInterface
from ..error import *

# Note:
#   - f stands for framework.
#   - Currently only tree-structure is implemented

class Node():
    """
    Framework Node:
      It consists of a framework instance and the next nodes.
      Currently it can only construct tree-shaped graph.
    """
    def __init__(self, f, prev, nexts=None):
        self.__name__ = 'node:{0}:{1}'.format(f.__name__,
                                              random.randint(0,1000))
        self._validate_f(f)
        self.f = f
        self.prev = prev
        if not nexts:
            self.nexts = nexts
        else:
            self.nexts = []

    def add_next(self, n):
        self.nexts.append(n)

    def _validate_f(self, f):
        """ Check if the framework implements all the necessary methods. """
        if f:
            attrs = dir(f)
            for method_name in PUBLIC_METHODS:
                assert method_name in attrs

class Workflow(object):
    """
    A Workflow instance contains the entire graph of frameworks.
    It iterates the graph each step.
    """
    def __init__(self, target_srcids, f_class_dict, f_graph_configs):
        """
        "f" stands for "framework" from now on.
        # Inputs:
        - target_srcids (list(str)) : The target srcids to parse/learn.
        - f_class_dict (dict): class_name: Class. E.g.,
            {
              "zodiac": ZodiacInterface
            }
        - f_graph_configs(dict): Configurations for the graph of the frameworks
                                 Each key is name of the framework.
                                 Each value is a tuple of configuration dict
                                 and child nodes.
            ex: {  # name          #configuration           #child nodes.
                    'zodiac': ({'target_building':xxx}, {'scrabble': ({}, [])})
                }
        """
        super(Workflow, self).__init__()
        self.target_srcids = target_srcids
        self.f_class_dict = f_class_dict

        # Instantiate dummy head node with the first configuration.
        base_config = list(f_graph_configs.values())[0][0]
        dummy_f = FrameworkInterface(**base_config)
        self.f_head = Node(dummy_f, None)

        # Instantiate the entire graph by recursive function call, init_node.
        f_nexts = []
        for f_name, f_graph_config in f_graph_configs.items():
            f_nexts.append(self.init_node(f_name, self.f_head, f_graph_config))
        self.f_head.nexts = f_nexts

    def init_node(self, f_name, prev, f_graph_configs):
        """
        Instantiate node and its children in a recursive manner.
        """
        f_config = f_graph_configs[0]
        f = self.f_class_dict[f_name](**f_config)
        curr_node = Node(f, prev)
        next_f_configs = f_graph_configs[1]
        nexts = []
        for next_f_name, next_f_config in next_f_configs.items():
            nexts.append(self.init_node(next_f_name, curr_node, next_f_config))
        curr_node.nexts = nexts
        return curr_node

    def select_informative_samples(self, sample_num):
        params = {
            'sample_num': sample_num
        }
        res_g = self._traverse_wrapper(self.f_head,
                                       'select_informative_samples',
                                       params)
        #TODO: Post processing the colleted result

    def predict_proba(self, target_srcids=None):
        params = {
            'target_srcids': target_srcisd
        }
        res_g = self._traverse_wrapper(self.f_head, 'predict_proba', params)
        # TODO: Post processing res_g to merge different results

    def predict(self, target_srcids):
        params = {
            'target_srcids': target_srcids
        }
        res_g = self._traverse_wrapper(self.f_head, ['predict'], [params])
        # TODO: Post processing res_g to merge different results

    def _traverse_wrapper(self, node, func_names, params, prev_attrs=[[]]):
        """
        Traversing the graph with the given jobs.
        At each node, it runs all the functions in func_names
        with param in params
        with reading prev_attr in prev_attrs.
        See update_model for an example.
        Outputs are flattened into a dictionary

        # Inputs:
        - node (Node): Current node to apply functions
                       and then its children recursively.
        - func_names (list(str)): list of functions to apply to a node.
        - params (list(dict)): list of param dicts for the functions above.

        """
        res_dict = {}
        for func_name, param, prev_attr in zip(func_names, params, prev_attrs):
            for attr in prev_attr:
                if node.prev:
                    param[attr] = getattr(node.prev.f, attr)
                else:
                    param[attr] = None
            func = getattr(node.f, func_name)
            try:
                res_dict[(str(node), func_name)] = func(**param)
            except EmptyTrainingSamples as e:
                print(e.msg)
            except:
                pdb.set_trace()

        for next_node in node.nexts:
            res_dict.update(self._traverse_wrapper(next_node, func_names,
                                                   params, prev_attrs))
        return res_dict

    def update_model(self, srcids):
        """
        Update model of each node. It consists of three steps for every node.
        First, update the model with given samples so far.
        Second, update the prior from the previous inferencer.
          Prior is the result prediction of the previous inferencer.
        Lastly, predict with the learned model to update its prediction.
          The predicted results may be used at the next node.

        # Input
          - srcids (list(str)): list of srcids to add.
        """
        params = [
            {
                'srcids': srcids,
            },
            {
            },
            {
                'target_srcids': self.target_srcids
            }
        ]
        func_names = ['update_model', 'update_prior', 'predict']
        prev_attrs = [[], ['pred_g'], []]
        self._traverse_wrapper(self.f_head, func_names, params, prev_attrs)

    def update_model_deprectaed(self, srcids):
        params = {
            'srcids': srcids
        }
        self._traverse_wrapper(self.f_head, ['update_model'], [params])
