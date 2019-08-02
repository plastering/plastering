import pdb

from mongoengine import *
import pprint
import pandas as pd
from tabulate import tabulate
pd.options.display.max_colwidth = 200
pp = pprint.PrettyPrinter(indent=2)

from .common import *

connect('plastering-withpg')


# Data Models

class RawMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    metadata = DictField()
    meta = {'allow_inheritance': True}

class LabeledMetadata(Document):
    srcid = StringField(required=True)
    building = StringField(required=True)
    fullparsing = DictField(default={})
    tagsets = ListField(StringField(), default=[])
    point_tagset = StringField(required=True)
    tagsets_parsing = DictField(default={})
    pgid = StringField()
    meta = {'allow_inheritance': True}


# Helper functions

def query_labels(pgid=None, **query):
    if pgid:
        return LabeledMetadata.objects(pgid=pgid, **query)
    else:
        return LabeledMetadata.objects(**query)

def print_rawmetadata(srcid, building):
    objs = RawMetadata.objects(srcid=srcid, building=building)
    metadata = objs[0].metadata
    df = pd.DataFrame(data=metadata, index=[srcid])
    df.index.name = 'srcid'
    print('Building: {0}'.format(building))
    print(tabulate(df, headers='keys', tablefmt='psql'))
    #print(df)

def print_fullparsing(srcid, building, pgid=None):
    fullparsing = query_labels(pgid=pgid,
                               srcid=srcid,
                               building=building,
                               ).first().fullparsing
    if not fullparsing:
        raise Exception('Full parsing is not registered yet for {0}'
                        .format(srcid))
    for metadata_type, parsed in fullparsing.items():
        print('In {0}'.format(metadata_type), end='')
        labels = [row[1] for row in parsed]
        labels = [label[2:] for label in labels]
        labels = [label for label in labels
                  if label and label not in ['O', 'leftidentifier',
                                             'rightidentifier', 'none']]
        new_labels = []
        for label in labels:
            if label not in new_labels:
                new_labels.append(label)
        print(new_labels)

def insert_groundtruth(srcid, building, pgid,
                       fullparsing=None, tagsets=None, point_tagset=None):
    obj = LabeledMetadata.objects(srcid=srcid, building=building, pgid=pgid)\
        .upsert_one(srcid=srcid, building=building, pgid=pgid)
    assert fullparsing or tagsets or point_tagset, 'WARNING:empty labels given'
    new_labels = {}
    if fullparsing:
        obj[FULL_PARSING] = fullparsing
    if point_tagset:
        obj[POINT_TAGSET] = point_tagset
    if tagsets:
        obj[ALL_TAGSETS] = tagsets
    obj.save()

def get_or_create(doc_type, **query):
    return doc_type.objects(**query).upsert_one(**query)
