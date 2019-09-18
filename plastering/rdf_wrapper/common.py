from rdflib import RDF, RDFS, OWL, URIRef, Namespace


def parse_srcid(point):
    return point.split('#')[-1]
