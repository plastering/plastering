import arrow

from plastering.inferencers.building_adapter_interface import BuildingAdapterInterface
from plastering.metadata_interface import *

'''
run "timeseries_init" first for each building used, to make sure the timeseries data is stored in DB
'''

target_building = 'ebu3b'
source_bulidings = ['ap_m']

labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

config = {
    'target_time_ranges': [
        (arrow.get(2016, 2, 1), arrow.get(2016, 2,6))
    ],
    'source_time_range': (arrow.get(2017, 1, 20), arrow.get(2017, 2,6)),
    'threshold': 0.99999999999
}

bl = BuildingAdapterInterface(target_building,
                              target_srcids,
                              source_bulidings,
                              config,
                              load_from_file=1)
#bl.run_auto()
srcids_labeled, tagset_preds, confidence = bl.predict()

