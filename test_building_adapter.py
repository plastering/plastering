from plastering.inferencers.building_adapter_interface import BuildingAdapterInterface
from plastering.metadata_interface import *

'''
run "timeseries_init" first for each building used, to make sure the timeseries data is stored in DB
'''

target_building = 'rice'
source_buliding = ['ucsd']

labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

bl = BuildingAdapterInterface(target_building, target_srcids, source_buliding)
bl.run_auto()
