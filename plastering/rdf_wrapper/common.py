from rdflib import RDF, RDFS, OWL, URIRef, Namespace

def parse_srcid(point):
    return point.split('#')[-1]

"""
def create_uri(name, ns=BASE):
    #return URIRef(ns + name)
    return BASE[name]
"""
