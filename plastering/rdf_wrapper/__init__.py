import os
from rdflib import RDF, RDFS, OWL, URIRef, Namespace
from . import rdflib_wrapper
from . import virtuoso_wrapper


TRIPLE_STORE_TYPE = os.environ.get('TRIPLE_STORE_TYPE', 'rdflib')
RDFLIB = 'rdflib'
VIRTUOSO = 'virtuoso'

if TRIPLE_STORE_TYPE == RDFLIB:
    from .rdflib_wrapper import *
elif TRIPLE_STORE_TYPE == VIRTUOSO:
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

class BrickGraph(object):
    def __init__(self,
                 empty=False,
                 version='1.0.2',
                 brick_file='brick/Brick_1_0_2.ttl',
                 brickframe_file='brick/BrickFrame_1_0_2.ttl',
                 triplestore_type=RDFLIB
                 ):
        if triplestore_type == RDFLIB:
            self.base_package = rdflib_wrapper
        elif triplestore_type == VIRTUOSO:
            self.base_package = virtuoso_wrapper
        self.g = self.base_package.init_graph(empty, brick_file,
                                              brickframe_file)

    def insert_point(self, name, tagset):
        return self.base_package.insert_point(self.g, name, tagset)

    def insert_triple(self, triple):
        return self.base_package.insert_triple(self.g, triple)

    def query_sparql(self, qstr):
        return self.base_package.query_sparql(self.g, qstr)

    def _try_add_pred_point_result(self, srcid, pred_point, pred_prob):
        triple = self._make_instance_tuple(srcid, pred_point)
        self.g.add(triple)
        self.pred_confidences[triple] = pred_prob

    def try_multiple_times(self, f, params):
        success = False
        for i in range(0, 10):
            try:
                res = f(**params)
                success = True
            except:
                print('WARNING: {0} temporarily failed'.format(str(f)))
            if success:
                break
            time.sleep(3)
        assert success, 'ERROR: {0} finally failed'.format(str(f))
        return res

    def add_pred_point_result(self,
                              srcid,
                              pred_point,
                              pred_prob):
        self.try_multiple_times(self._try_add_pred_point_result, {
            'srcid': srcid,
            'pred_point': pred_point,
            'pred_prob': pred_prob
        })

    def get_vavs(self):
        qstr = """
        select ?vav where {{
          ?vav a/rdfs:subClassOf* brick:vav .
          }}
        """
        res = self.query_sparql(qstr)
        vavs = [row['vav'] for row in res]
        return vavs

    def get_vav_points(self, vav):
        qstr = """
        select ?point where {{
        ?point bf:isPointOf {0}.
        }}
        """.format(vav.n3())
        res = self.query_sparql(qstr)
        points = [row['point'] for row in res]
        return points
