import os
from rdflib import RDF, RDFS, OWL, URIRef, Namespace


TRIPLE_STORE_TYPE = os.environ.get('TRIPLE_STORE_TYPE', 'rdflib')

if TRIPLE_STORE_TYPE == 'rdflib':
    from .rdflib_wrapper import *
elif TRIPLE_STORE_TYPE == 'virtuoso':
    from .virtuoso_wrapper import *
else:
    raise Exception('Triple store type not defined for: {0}'
                    .format(TRIPLE_STORE_TYPE))


def get_top_class(point_tagset):
    if isinstance(point_tagset, URIRef):
        base_class = point_tagset.split('#')[-1].split('_')[-1]
    elif isinstance(point_tagset, str):
        base_class = point_tagset.split('_')[-1]
        point_tagset = BRICK[point_tagset]
    else:
        raise Exception('Behavior not defined for {0}'.format(point_tagset))
    base_tagset = BRICK[base_class]
    if base_class in ['sensor', 'meter']:
        qstr = """
        select ?superclass where {{
        ?superclass rdfs:subClassOf {0}.
        {1} rdfs:subClassOf* ?superclass.
        }}
        """.format(base_tagset.n3(), point_tagset.n3())
        res = query_sparql(schema_g, qstr)
        assert res, 'No super class found for {0}'.format(point_tagset)
        base_class = res[0]['superclass'].split('#')[-1]
    return base_class

def get_point_type(g, point):
    qstr = """
    select ?t where {{
    {0} a ?t .
    }}
    """.format(point.n3())
    res = query_sparql(g, qstr)
    t = res[0]['t']
    return t.split('#')[-1]
    #return t

def get_vav_points(g, vav):
    qstr = """
    select ?point where {{
    ?point bf:isPointOf {0}.
    }}
    """.format(vav.n3())
    res = query_sparql(g, qstr)
    points = [row['point'] for row in res]
    return points


def get_vavs(g):
    qstr = """
    select ?vav where {{
      ?vav a/rdfs:subClassOf* brick:vav .
      }}
    """
    res = query_sparql(g, qstr)
    vavs = [row['vav'] for row in res]
    return vavs


def get_instance_tuples(g):
    qstr = sparql_prefix + """
    select ?s ?o where {
        ?s a ?o.
        FILTER(STRSTARTS(STR(?s), "%s"))
    }
    """ % (BASE) # Query selecting any instances with name space BASE.
    res = g.query(qstr)
    if TRIPLE_STORE_TYPE == 'virtuoso':
        return {row['s'].split('#')[-1]: row['o'].split('#')[-1] for row in res}
    elif TRIPLE_STORE_TYPE == 'rdflib':
        # TODO: Need to validate this
        return {row[0].split('#')[-1]: row[1].split('#')[-1] for row in res}
