import sys
import os
import json
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/..')
from plastering.inferencers.active_learning_interface import ActiveLearningInterface
from plastering.metadata_interface import *

target_building = sys.argv[1]
try:
    source_building = sys.argv[2]
except:
    source_building = None

labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

fold = 10
rounds = 250

EXP_NUM = 4

for exp_id in range(0, EXP_NUM):
    al = ActiveLearningInterface(
        target_building=target_building,
        target_srcids=target_srcids,
        fold=fold,
        rounds=rounds,
        source_building=source_building,
        )

    al.learn_auto()
    acc_sum = [np.nanmean(i) for i in al.learner.acc_sum]
    f1_micro_sum = [np.nanmean(i) for i in al.learner.f1_micro_sum]
    f1_macro_sum = [np.nanmean(i) for i in al.learner.f1_macro_sum]

    results = zip(acc_sum, f1_micro_sum, f1_macro_sum)
    outputs = [{ 'metrics': { 'accuracy': acc, 'f1_micro': f1_micro, 'f1_macro':f1_macro }, \
        # to Jason: revert the name
                'learning_srcids': i+1 } \
                for i,(acc,f1_micro,f1_macro) in enumerate(results) \
                ]
    if source_building:
        outputfile = 'result/pointonly_transfer_{0}_{1}_{2}_{3}.json'\
            .format('al_hong', target_building, source_building, exp_id)
    else:
        outputfile = 'result/pointonly_transfer_{0}_{1}_{2}.json'\
            .format('al_hong', target_building, exp_id)

    with open(outputfile, 'w') as fp:
        json.dump(outputs, fp)
