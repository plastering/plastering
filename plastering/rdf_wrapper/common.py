from rdflib import RDF, RDFS, OWL, URIRef, Namespace

BRICK_VERSION = '1.0.2'
BRICK = Namespace('https://brickschema.org/schema/{0}/Brick#'.format(BRICK_VERSION))
BF = Namespace('https://brickschema.org/schema/{0}/BrickFrame#'.format(BRICK_VERSION))
BASE = Namespace('http://example.com#')
sparql_prefix = """
prefix brick: <{0}>
prefix rdf: <{1}>
prefix rdfs: <{2}>
prefix base: <{3}>
prefix bf: <{4}>
prefix owl: <{5}>
""".format(str(BRICK), RDF, RDFS, BASE, str(BF), OWL)

BRICK_DIR = '/home/jbkoh/repo/plastering_merged/brick/'

BRICK_FILE = BRICK_DIR + '/Brick_{0}.ttl'\
    .format(BRICK_VERSION.replace('.', '_'))
BF_FILE = BRICK_DIR + '/BrickFrame_{0}.ttl'\
    .format(BRICK_VERSION.replace('.', '_'))


def insert_triple(g, triple):
    g._create_insert_query([triple])

def parse_srcid(point):
    return point.split('#')[-1]

def create_uri(name, ns=BASE):
    #return URIRef(ns + name)
    return BASE[name]
