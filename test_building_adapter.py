from plastering.inferencers.building_adapter_interface import BuildingAdapterInterface
from plastering.metadata_interface import *

target_building = 'rice'
source_buliding = ['ucsd']

labeled_list = LabeledMetadata.objects(building=target_building)
target_srcids = [labeled['srcid'] for labeled in labeled_list]

bl = BuildingAdapterInterface(target_building, target_srcids, source_buliding)
bl.run_auto()
