import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL
import pdb
from functools import reduce

adder = lambda x,y: x+y

BRICK = Namespace('https://brickschema.org/schema/1.0.2/Brick#')
BF = Namespace('https://brickschema.org/schema/1.0.2/BrickFrame#')
SKOS = Namespace('http://www.w3.org/2004/02/skos/core#')

def get_all_tagsets(g):
    qstr = """
    select ?s where {
    {{?s rdfs:subClassOf* brick:Location.}
    UNION
    {?s rdfs:subClassOf* brick:Equipment.}}
    UNION
    {?s rdfs:subClassOf* brick:Point.}
    }
    """
    res = g.query(qstr, initNs={'brick': BRICK, 'rdfs': RDFS})
    return [row['s'] for row in res]

def lowercase_tagset(g, tagset):
    lowered_tagset = BRICK[tagset.split('#')[-1].lower()]

    qstr1 = """
    select ?s ?p where {{
    ?s ?p {0}.
    }}
    """.format(tagset.n3())
    res1 = g.query(qstr1)
    # Replace triples
    for row in res1:
        g.remove((row[0], row[1], tagset))
        g.add((row[0], row[1], lowered_tagset))

    qstr2 = """
    select ?p ?o where {{
    {0} ?p ?o.
    }}
    """.format(tagset.n3())
    res2 = g.query(qstr2)
    # Replace triples
    for row in res2:
        g.remove((tagset, row[0], row[1]))
        g.add((lowered_tagset, row[0], row[1]))

def lowercase_schema():
    g = Graph()
    g.parse('./Brick_1_0_2.original.ttl', format='turtle')
    tagsets = get_all_tagsets(g)
    """
    new_g = Graph()
    new_g.bind('brick', BRICK)
    new_g.bind('bf', BF)
    new_g.bind('rdfs', RDFS)
    new_g.bind('rdf', RDF)
    new_g.bind('owl', OWL)
    new_g.bind('skos', SKOS)
    """
    for tagset in tagsets:
        lowercase_tagset(g, tagset)
    g.serialize('./Brick_1_0_2.lower.ttl', format='turtle')

def lowercase_building(filename):
    g = Graph()
    g.parse('./Brick_1_0_2.original.ttl', format='turtle')
    tagsets = get_all_tagsets(g)
    building_g = Graph()
    building_g.parse(filename, format='turtle')
    for tagset in tagsets:
        lowercase_tagset(building_g, tagset)
    building_g.serialize('./test.ttl', format='turtle')

if __name__ == '__main__':
    #lowercase_building('ebu3b_brick.ttl')
    lowercase_schema()
