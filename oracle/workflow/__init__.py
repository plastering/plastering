import pdb
from ..frameworks.framework_interface import PUBLIC_METHODS
from ..frameworks.framework_interface import FrameworkInterface
from ..error import *

# Note:
#   - f stands for framework.
#   - Currently only tree-structure is implemented

class Node():
    """docstring for Node"""
    def __init__(self, f, prev_f, nexts=None):
        self._validate_f(f)
        self.f = f
        self.prev_f = prev_f
        #self.g = rdf.Graph() # TODO: implement
        if not nexts:
            self.nexts = nexts
        else:
            self.nexts = []

    def add_next(self, n):
        self.nexts.append(n)

    def _validate_f(self, f):
        if f:
            attrs = dir(f)
            for method_name in PUBLIC_METHODS:
                assert method_name in attrs

class Workflow(object):
    """docstring for Workflow"""
    def __init__(self, target_srcids, f_class_dict, f_graph_configs):
        """
        inputs
        f_names (dict): graph of the frameworks (name + configuration)
            ex: {
                    "zodiac": ({}, {
                        "scrabble": ({}, [])
                    })
                }
        """
        super(Workflow, self).__init__()
        self.target_srcids = target_srcids
        self.f_class_dict = f_class_dict
        # TODO: Below is for testing.
        #       Dummy node should be generated programatically.
        """
        base_config = {
            'target_building': None,
            'target_srcids': [],
            'source_buildings': None
        }
        """
        base_config = list(f_graph_configs.values())[0][0]
        dummy_f = FrameworkInterface(**base_config)
        self.f_head = Node(dummy_f, None)
        f_nexts = []
        for f_name, f_graph_config in f_graph_configs.items():
            f_nexts.append(self.init_node(f_name, self.f_head, f_graph_config))
        self.f_head.nexts = f_nexts

    def init_node(self, f_name, prev_f, f_graph_configs):
        f_config = f_graph_configs[0]
        f = self.f_class_dict[f_name](**f_config)
        curr_node = Node(f, prev_f)
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
        res_g = self._traverse_wrapper(self.f_head, 'select_informative_samples', params)
        #TODO: Post processing the colleted result


    # ESSENTIAL
    def predict_proba(self, target_srcids=None):
        # TODO
        params = {
            'target_srcids': target_srcisd
        }
        # TODO: Below needs to use the outputs of the previous results
        res_g = self._traverse_wrapper(self.f_head, 'predict_proba', params)

    # ESSENTIAL
    def predict(self, target_srcids):
        # TODO
        """
        """
        params = {
            'target_srcids': target_srcids
        }
        # TODO: Below needs to use the outputs of the previous results
        res_g = self._traverse_wrapper(self.f_head, ['predict'], [params])
        # TODO: Post processing res_g to merge different results

    def _traverse_wrapper(self, node, func_names, params, prev_attrs=[[]]):
        res_dict = {}
        for func_name, param, prev_attr in zip(func_names, params, prev_attrs):
            for attr in prev_attr:
                if node.prev_f:
                    param[attr] = getattr(node.prev_f.f, attr)
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


    def _traverse_wrapper_dp(self, node, func_name, params, prev_attrs=[]):
        # TOOD: Augment this if we need more than tree
        """
        params
        """
        for prev_attr in prev_attrs:
            params[prev_attr] = getattr(node.prev_f, prev_attr)
        func = getattr(node.f, func_name)
        try:
            func(**params)
        except EmptyTrainingSamples as e:
            print(e.msg)
        res_dict = {}
        for next_node in node.nexts:
            res = self._traverse_wrapper(next_node, func_name,
                                         params, prev_attrs)
            res_dict[str(next_node)] = res
        return res_dict

    def update_model(self, srcids):
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

    def update_model_dep(self, srcids):
        params = {
            'srcids': srcids
        }
        self._traverse_wrapper(self.f_head, ['update_model'], [params])

    def select_informative_samples(self, sample_num):
        pass


