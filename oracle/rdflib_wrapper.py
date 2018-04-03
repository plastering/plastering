import rdflib
from rdflib import Graph, RDF, RDFS, OWL

import pdb


BRICK = 'https://brickschema.org/schema/1.0.1/Brick#'
BF = 'https://brickschema.org/schema/1.0.1/BrickFrame#'
BASE = 'http://example.com#'

sparql_prefix = """
prefix brick: <{0}>
prefix rdf: <{1}>
prefix rdfs: <{2}>
prefix base: <{3}>
prefix bf: <{4}>
prefix owl: <{5}>
""".format(BRICK, RDF, RDFS, BASE, BF, OWL)


def init_graph():
    g = Graph()
    g.parse('Brick/dist/Brick.ttl', format='turtle')
    g.parse('Brick/dist/BrickFrame.ttl', format='turtle')
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
    
