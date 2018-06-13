import sys, os
import re
import pdb
import json
import rdflib
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/..')
from shutil import copyfile

from plastering.inferencers.zodiac_new import ZodiacInterface
from plastering.metadata_interface import *
from plastering.rdf_wrapper import *

BRICK_VERSION_STR = BRICK_VERSION.replace('.', '_')
brick_dir = dir_path + '/../brick/'

base_filename = 'Brick_{0}.ttl.\\d+'.format(BRICK_VERSION_STR)
target_file = brick_dir + 'Brick_{0}.ttl'.format(BRICK_VERSION_STR)
source_file = brick_dir + 'Brick_{0}.lower.ttl'.format(BRICK_VERSION_STR)

files = os.listdir(brick_dir)
file_versions = []
for fname in files:
    if re.match(base_filename, fname):
        file_versions.append(int(fname.split('.')[-1]))
save_flag = input("Do you want backup the file? (yes/no)")
if save_flag == "yes":
    new_version = max(file_versions) + 1
    copyfile(target_file, target_file + '.' + str(new_version))
elif save_flag == "no":
    pass
else:
    raise Exception("yes or no?")

g = rdflib.Graph()
g.parse(source_file, format='turtle')

# Add new tagsets to Brick.ttl
triples = [
    (BRICK['heating_cooling_status'], RDFS.subClassOf, BRICK['status']),
    (BRICK['heating_cooling_status'], RDF.type, OWL.Class),
    (BRICK['max_air_flow_sensor'], RDFS.subClassOf, BRICK['air_flow_sensor']),
    (BRICK['max_air_flow_sensor'], RDF.type, OWL.Class),
    (BRICK['valve_position_sensor'], RDFS.subClassOf, BRICK['sensor']),
    (BRICK['valve_position_sensor'], RDF.type, OWL.Class),
    (BRICK['water_temperature_sensor'], RDFS.subClassOf, BRICK['temperature_sensor']),
    (BRICK['water_temperature_sensor'], RDF.type, OWL.Class),
    (BRICK['chilled_water_temperature_sensor'], RDFS.subClassOf, BRICK['water_temperature_sensor']),
    (BRICK['chilled_water_temperature_sensor'], RDF.type, OWL.Class),
    (BRICK['steam_temperature_sensor'], RDFS.subClassOf, BRICK['temperature_sensor']),
    (BRICK['steam_temperature_sensor'], RDF.type, OWL.Class),
]

for triple in triples:
    g.add(triple)
g.bind('brick', BRICK)
g.bind('bf', BF)
g.bind('rdf', RDF)
g.bind('rdfs', RDFS)
g.serialize(target_file, format='turtle')


# Add the tagsets to point_tagsets.json
with open(brick_dir + '/point_tagsets.json', 'r') as fp:
    point_tagsets = json.load(fp)
point_tagsets.append('heating_cooling_status')
point_tagsets.append('max_air_flow_sensor')
point_tagsets.append('valve_position_sensor')
point_tagsets.append('chilled_water_temperature_sensor')
point_tagsets.append('steam_temperature_sensor')
with open(brick_dir + '/point_tagsets.json', 'w') as fp:
    json.dump(point_tagsets, fp)

def find_root(tree, routes):
    root = tree[routes[0]]
    new_root = root
    for route in routes[1:]:
        found = False
        for branch in root:
            if route in branch.keys():
                new_root = branch[route]
                found = True
                break
        assert found, 'routes are incorrect at {0} in {1}'\
            .format(route, routes)
        root = new_root
    return root


# Add the tagsets to the tree
with open(brick_dir + '/tagset_tree.json', 'r') as fp:
    tagset_tree = json.load(fp)

root = find_root(tagset_tree, ['status'])
root.append(
    {
        'heating_cooling_status': []
    }
)
root = find_root(tagset_tree, ['sensor', 'temperature_sensor'])
root.append(
    {'water_temperature_sensor': [
        {
            'chilled_water_temperature_sensor': []
        }
    ]}
)
root.append({
    'steam_temperature_sensor': []
})
root = find_root(tagset_tree, ['sensor'])
root.append(
    {
        'valve_position_sensor': []
    }
)
root = find_root(tagset_tree, ['sensor', 'flow_sensor', 'air_flow_sensor'])
root.append(
    {
        'max_air_flow_sensor': []
    }
)
with open(brick_dir + '/tagset_tree.json', 'w') as fp:
    json.dump(tagset_tree, fp)

# Add the tagsets to the subclasses
with open(brick_dir + '/point_subclass_dict.json', 'r') as fp:
    point_subclasses = json.load(fp)
point_subclasses['status'].append('heating_cooling_status')
point_subclasses['sensor'].append('valve_position_sensor')
point_subclasses['air_flow_sensor'].append('max_air_flow_sensor')
point_subclasses['temperature_sensor'].append('water_temperature_sensor')
point_subclasses['temperature_sensor'].append('steam_temperature_sensor')
point_subclasses['water_temperature_sensor'] = ['chilled_water_temperature_sensor']
with open(brick_dir + '/point_subclass_dict.json', 'w') as fp:
    json.dump(point_subclasses, fp)

