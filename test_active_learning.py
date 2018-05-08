from plastering.inferencers.active_learning_interface import ActiveLearningInterface
from plastering.metadata_interface import *

target_building = 'uva_cse'

labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

fold = 10
rounds = 100
al = ActiveLearningInterface(
    target_building,
    target_srcids,
    fold=fold,
    rounds=rounds
    )

al.run_auto()

