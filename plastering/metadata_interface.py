import pdb

from mongoengine import *
import pprint
import pandas as pd
from tabulate import tabulate
pd.options.display.max_colwidth = 200
pp = pprint.PrettyPrinter(indent=2)

connect('plastering')


# Data Models

class RawMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    metadata = DictField()

class LabeledMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    fullparsing = DictField()
    tagsets = ListField(StringField())
    point_tagset = StringField(required=True)
    tagsets_parsing = DictField()

# Helper functions

def print_rawmetadata(srcid, building):
    objs = RawMetadata.objects(srcid=srcid, building=building)
    metadata = objs[0].metadata
    df = pd.DataFrame(data=metadata, index=[srcid])
    df.index.name = 'srcid'
    print('Building: {0}'.format(building))
    print(tabulate(df, headers='keys', tablefmt='psql'))
    #print(df)

def insert_groundtruth(srcid, building,
                       fullparsing=None, tagsets=None, point_tagset=None):
    obj = LabeledMetadata.objects(srcid=srcid)\
        .upsert_one(srcid=srcid, building=building)
    assert fullparsing or tagsets or point_tagset, 'WARNING:empty labels given'
    new_labels = {}
    if fullparsing:
        new_labels['set__fullparsing'] = fullparsing
    if point_tagset:
        new_labels['set__point_tagset'] = point_tagset
    if tagsets:
        new_labels['set__tagsets'] = tagsets
    obj.update(**new_labels, upsert=True)
