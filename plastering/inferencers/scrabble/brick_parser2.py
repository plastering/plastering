import rdflib
import pdb
from rdflib import Namespace
import json
import os
from collections import defaultdict


schema_g = rdflib.Graph()

# TODO: Migrate everyhting here under rdf_wrapper


def get_prefixes(version):
    BRICK = Namespace('https://brickschema.org/schema/{0}/Brick#'.format(version))
    BRICKFRAME = Namespace('https://brickschema.org/schema/{0}/BrickFrame#'.format(version))
    BRICKTAG = Namespace('https://brickschema.org/schema/{0}/BrickTag#'.format(version))
    return {
        'brick': BRICK,
        'bf': BRICKFRAME,
        'tag': BRICKTAG,
        'rdfs': rdflib.RDFS,
        'rdf': rdflib.RDF,
        'owl': rdflib.OWL,
    }


def get_schema_graph(version):
    if not schema_g:
        schema_g.parse('https://github.com/BrickSchema/Brick/releases/download/v{0}/Brick.ttl'.format(version), format='turtle')
        schema_g.parse('https://github.com/BrickSchema/Brick/releases/download/v{0}/BrickFrame.ttl'.format(version), format='turtle')
        # TODO: Parameterize this or use an online link.
    return schema_g

def extract_tagset(uri):
    return uri.split('#')[-1].lower()

def get_subclasses(version, topclass):
    qstr = """
    SELECT ?point where {{
    ?point rdfs:subClassOf+ {0}.
    }}
    """.format(topclass)
    g = get_schema_graph(version)
    return [extract_tagset(row[0]) for row in g.query(qstr, initNs=get_prefixes(version))]

def get_direct_subclasses_dict(version, topclass='bf:TagSet'):
    qstr = """
    SELECT ?child ?parent where {{
    ?child rdfs:subClassOf ?parent.
    }}
    """.format(topclass)
    g = get_schema_graph(version)
    subclasses = defaultdict(set)
    for [child, parent] in g.query(qstr, initNs=get_prefixes(version)):
        subclasses[extract_tagset(parent)].add(extract_tagset(child))
    for k, v in subclasses.items():
        subclasses[k] = list(v)
    return subclasses

def get_subclasses_dict(version, topclass='bf:TagSet'):
    # NOTE: Maybe this should consider equivalentClassOF too.
    qstr = """
    SELECT ?child ?parent where {{
    ?child rdfs:subClassOf+ ?parent.
    ?parent rdfs:subClassOf* {0}.
    }}
    """.format(topclass)
    g = get_schema_graph(version)
    subclasses = defaultdict(set)
    for [child, parent] in g.query(qstr, initNs=get_prefixes(version)):
        subclasses[extract_tagset(parent)].add(extract_tagset(child))
    for k, v in subclasses.items():
        subclasses[k] = list(v)
    return subclasses

pointPostfixes = ['alarm', 'sensor', 'setpoint', 'command', 'status', 'meter']
equipPostfixes = ['system', 'dhws', 'tower', 'chiller', 'coil', 'fan',
                       'hws', 'storage', 'battery', 'condenser', 'unit', 'fcu',
                       'vav', 'volume', 'economizer', 'hood', 'filter', 'vfd',
                       'valve', 'condensor', 'damper', 'hx', 'exchanger',
                       'thermostat', 'ahu', 'drive', 'heater', 'pump',
                       'conditioning', 'ws', 'dhws', 'elevator', 'fcp',
                       'panel', 'weather', 'generator', 'inverter', 'response',
                       'cws', 'crac', 'equipment', 'hvac']


def construct_subclass_tree(head, tagset_type, subclasses_dict):
    upper_tagset = head.split(':')[-1].lower()
    #res = g.query(directSubclassesQuery(head))
    try:
        res = subclasses_dict[upper_tagset]
    except:
        pdb.set_trace()
    subclasses = list()
    tagsets = list()
    branches = list()
    for row in res:
        thing = row[0]
        subclass = thing.split('#')[-1]
        tagset = subclass.lower()
        if tagset_type == 'point' and tagset.split('_')[-1]\
           not in pointPostfixes:
            continue
        if tagset_type == 'equip' and tagset.split('_')[-1] \
           not in equipPostfixes:
            continue

        subclasses.append(subclass)
        tagsets.append(tagset)
        branches.append(construct_subclass_tree('brick:'+subclass, tagset_type, subclasses_dict))
    tree = {upper_tagset: branches}
    return tree

def get_tagset_tree(version):
    subclasses = get_direct_subclasses_dict(version)
    tagsetTree = dict()
    for head in ['Sensor', 'Alarm', 'Status', 'Setpoint', 'Command', 'Meter']:
        tagsetTree.update(construct_subclass_tree('brick:'+head, 'point', subclasses))
    for head in ['Equipment']:
        tagsetTree.update(construct_subclass_tree('brick:'+head, 'equip', subclasses))
    for head in ['Location']:
        tagsetTree.update(construct_subclass_tree('brick:'+head, 'location', subclasses))



