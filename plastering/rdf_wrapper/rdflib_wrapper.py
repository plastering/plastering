import pdb

import rdflib
from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace
from copy import deepcopy

from .common import *


preloaded_g = None
schema_g = None
empty_g = Graph()

def adder(x, y):
    return x + y

def init_graph(empty=False, brick_file=None, brickframe_file=None):
    global schema_g
    global preloaded_g
    if schema_g == None:
        schema_g = Graph()
        schema_g.parse(brick_file, format='turtle')
        schema_g.parse(brickframe_file, format='turtle')

    if empty:
        return deepcopy(empty_g)
    else:
        if preloaded_g == None:
            preloaded_g = Graph()
            preloaded_g.parse(brick_file, format='turtle')
            preloaded_g.parse(brickframe_file, format='turtle')
        return deepcopy(preloaded_g)
    return g

def insert_point(g, name, tagset):
    triple = (URIRef(name), RDF.type, BRICK[tasget])
    g.add(triple)

def insert_triple(g, triple):
    g.add(triple)

def query_sparql(g, qstr):
    res = (g + schema_g).query(qstr).bindings
    return res

