from plastering.rdflib_wrapper import *
from plastering.metadata_interface import *



print(get_top_class('zone_temperature_sensor'))

building = 'sdh'
for obj in LabeledMetadata.objects(building=building):
    point_tagset = obj.point_tagset
    print(point_tagset)
