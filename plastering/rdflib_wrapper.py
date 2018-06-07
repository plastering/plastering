import rdflib
from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace
from copy import deepcopy

import pdb


BRICK_VERSION = '1.0.2'

#BRICK = 'https://brickschema.org/schema/1.0.1/Brick#'
#BF = 'https://brickschema.org/schema/1.0.1/BrickFrame#'
# TODO: These should be NS instead of raw strings.
BRICK = Namespace('https://brickschema.org/schema/{0}/Brick#'.format(BRICK_VERSION))
BF = Namespace('https://brickschema.org/schema/{0}/BrickFrame#'.format(BRICK_VERSION))
#BF = 'https://brickschema.org/schema/{0}/BrickFrame#'.format(BRICK_VERSION)
BASE = 'http://example.com#'

sparql_prefix = """
prefix brick: <{0}>
prefix rdf: <{1}>
prefix rdfs: <{2}>
prefix base: <{3}>
prefix bf: <{4}>
prefix owl: <{5}>
""".format(str(BRICK), RDF, RDFS, BASE, str(BF), OWL)



preloaded_g = Graph()
preloaded_g.parse('brick/Brick_{0}.ttl'
            .format(BRICK_VERSION.replace('.', '_')), format='turtle')
preloaded_g.parse('brick/BrickFrame_{0}.ttl'
            .format(BRICK_VERSION.replace('.', '_')), format='turtle')
empty_g = Graph()
schema_g = deepcopy(preloaded_g)

def adder(x, y):
    return x + y


def init_graph(empty=False):
    if empty:
        return deepcopy(empty_g)
    else:
        return deepcopy(preloaded_g)
    return g

def get_instance_tuples(g):
    qstr = sparql_prefix + """
    select ?s ?o where {
        ?s a ?o.
        FILTER(STRSTARTS(STR(?s), "%s"))
    }
    """ % (BASE) # Query selecting any instances with name space BASE.
    res = g.query(qstr)
    return {row[0].split('#')[-1]: row[1].split('#')[-1] for row in res}

def insert_point(g, name, tagset):
    triple = (URIRef(name), RDF.type, BRICK[tasget])
    g.add(triple)

def insert_triple(g, triple):
    g.add(triple)

def create_uri(name, ns=BASE):
    return URIRef(ns + name)

def query_sparql(g, qstr):
    qstr = sparql_prefix + qstr
    res = (g + schema_g).query(qstr).bindings
    return res

def get_vavs(g):
    qstr = """
    select ?vav where {
      ?vav a/rdfs:subClassOf* brick:VAV .
      }
    """
    res = query_sparql(g, qstr)
    vavs = [row['vav'] for row in res]
    return vavs

def get_vav_points(g, vav):
    qstr = """
    select ?point where {
    ?point bf:isPointOf {0}.
    ?point a/rdfs:subClassOf brick:Point.
    }
    """.format(vav.n3())
    res = query_sparql(g, qstr)
    points = [row['point'] for row in res]
    return points

def get_point_type(g, point):
    qstr = """
    select ?t {
    {0} a ?t .
    }
    """
    res = query_sparql(g, qstr)
    t = res[0]['t']
    return t

