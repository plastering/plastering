import pdb

import rdflib
from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace
from copy import deepcopy

from .common import *


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

def insert_point(g, name, tagset):
    triple = (URIRef(name), RDF.type, BRICK[tasget])
    g.add(triple)

def insert_triple(g, triple):
    g.add(triple)

def query_sparql(g, qstr):
    qstr = sparql_prefix + qstr
    res = (g + schema_g).query(qstr).bindings
    return res

