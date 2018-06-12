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
from plastering.rdflib_wrapper import *

BRICK_VERSION_STR = BRICK_VERSION.replace('.', '_')
brick_dir = dir_path + '/../brick/'

base_filename = 'Brick_{0}.ttl.\\d+'.format(BRICK_VERSION_STR)
target_file = brick_dir + 'Brick_{0}.ttl'.format(BRICK_VERSION_STR)

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

# Add new tagsets to Brick.ttl
triples = [
    (BRICK['heating_cooling_status'], RDFS.subClassOf, BRICK['status']),
    (BRICK['heating_cooling_status'], RDF.type, OWL.Class)
]


# Add the tagsets to point_tagsets.json
with open(brick_dir + '/point_tagsets.json', 'r') as fp:
    point_tagsets = json.load(fp)
point_tagsets.append('heating_cooling_status')
with open(brick_dir + '/point_tagsets.json', 'w') as fp:
    json.dump(point_tagsets, fp)


# Add the tagsets to the tree
with open(brick_dir + '/tagset_tree.json', 'r') as fp:
    tagset_tree = json.load(fp)
tagset_tree['status'].append(
    {
        'heating_cooling_status': []
    }
)
with open(brick_dir + '/tagset_tree.json', 'w') as fp:
    json.dump(tagset_tree, fp)

# Add the tagsets to the subclasses
with open(brick_dir + '/point_subclass_dict.json', 'r') as fp:
    point_subclasses = json.load(fp)
point_subclasses['status'].append('heating_cooling_status')
with open(brick_dir + '/point_subclass_dict.json', 'w') as fp:
    json.dump(point_subclasses, fp)

