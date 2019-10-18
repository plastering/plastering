import pdb
from copy import deepcopy
import logging

import arrow

#from char2ir import char2ir_onestep
from ir2tagsets import ir2tagset_onestep
from common import *


def char2tagset_onestep(step_data,
                        building_list,
                        source_sample_num_list,
                        target_building,
                        use_cluster_flag=False,
                        use_brick_flag=False,
                        crftype='crfsuite',
                        eda_flag=False,
                        negative_flag=True,
                        debug_flag=True,
                        n_jobs=8, # TODO parameterize
                        ts_flag=False,
                        inc_num=10,
                        crfqs='confidence',
                        entqs='phrase_util'
                        ):
    begin_time = arrow.get()
    step_data = deepcopy(step_data)
    step_data['learning_srcids'] = step_data['next_learning_srcids']

    step_data = char2ir_onestep(step_data,
                                     building_list,
                                     source_sample_num_list,
                                     target_building,
                                     use_cluster_flag,
                                     use_brick_flag,
                                     crftype,
                                     inc_num / 2,
                                     crfqs)

    pdb.set_trace()
    step_data = ir2tagset_onestep(step_data,
                                  building_list,
                                  source_sample_num_list,
                                  target_building,
                                  use_cluster_flag,
                                  use_brick_flag,
                                  eda_flag,
                                  negative_flag,
                                  debug_flag,
                                  n_jobs, # TODO parameterize
                                  ts_flag,
                                  inc_num / 2,
                                  entqs)
    end_time = arrow.get()
    logging.info('An iteration takes ' + str(end_time - begin_time))
    return step_data

    
def char2tagset_iteration(iter_num, custom_postfix='', *params):
    """
    params: 
        building_list,
        source_sample_num_list,
        target_building,
        use_cluster_flag=False,
        use_brick_flag=False,
        crftype='crfsuite'
        eda_flag=False,
        negative_flag=True,
        debug_flag=True,
        n_jobs=8, # TODO parameterize
        ts_flag=False)
    """
    begin_time = arrow.get()
    building_list = params[0]
    source_sample_num_list = params[1]
    prev_data = {'iter_num':0,
                 'next_learning_srcids': get_random_srcids(
                                        building_list,
                                        source_sample_num_list),
                 'model_uuid': None}
    step_datas = iteration_wrapper(iter_num, char2tagset_onestep, 
                                   prev_data, *params)

    building_list = params[0]
    target_building = params[2]
    postfix = 'char2tagset_iter' 
    if custom_postfix:
        postfix += '_' + custom_postfix
    with open('result/crf_entity_iter_{0}_{1}.json'\
            .format(''.join(building_list+[target_building]), postfix), 'w') as fp:
        json.dump(step_datas, fp, indent=2)
    end_time = arrow.get()
    print(iter_num, " iterations took: ", end_time - begin_time)
