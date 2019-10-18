from copy import deepcopy
import pdb
from uuid import uuid4 as gen_uuid

import rdflib
from rdflib import RDFS, RDF, OWL, Namespace
from rdflib.namespace import FOAF
from SPARQLWrapper import SPARQLWrapper
from SPARQLWrapper import JSON, SELECT, INSERT, DIGEST, GET, POST
from rdflib import URIRef, Literal

from .common import *
from ..helpers import chunks


def init_graph(empty=False):
    be = BrickEndpoint('http://localhost:8890/sparql',
                       BRICK_VERSION,
                       #load_schema=True #TODO: temporarily force to enable
                       load_schema=empty,
                       )

    return be



def query_sparql(g, qstr):
    res = g.raw_query(qstr)
    return res


class BrickEndpoint(object):

    def __init__(self, sparql_url, brick_version, base_ns='', load_schema=True):
        BRICK_VERSION = brick_version
        self.sparql_url = sparql_url
        self.sparql = SPARQLWrapper(endpoint=self.sparql_url,
                                    updateEndpoint=self.sparql_url + '-auth')
        self.sparql.queryType= SELECT
        self.sparql.setCredentials('dba', 'dba')
        self.sparql.setHTTPAuth(DIGEST)
        if not base_ns:
            base_ns = 'http://example.com/'
        self.base_graph = 'urn:' + str(gen_uuid())
        self.sparql.addDefaultGraph(self.base_graph)
        self.namespaces = {
            '': BASE,
            'brick':BRICK,
            'bf': BF,
            'rdfs': RDFS,
            'rdf': RDF,
            'owl': OWL,
            'foaf': FOAF
        }
        sparql_prefix = ''
        #for prefix, ns in self.namespaces.items():
        #    ns_n3 = ns.uri.n3()
        #    sparql_prefix += 'prefix {0}: {1}\n'.format(prefix, ns_n3)
        #sparql_prefix += '\n'

        self._init_brick_constants()
        if load_schema:
            self.load_schema()

    def _init_brick_constants(self):
        self.HAS_LOC = URIRef(BF + 'hasLocation')

    def _get_sparql(self):
        # If need to optimize accessing sparql object.
        return self.sparql

    def update(self, qstr):
        return self.query(qstr, is_update=True)

    def _format_select_res(self, raw_res):
        var_names = raw_res['head']['vars']
        values = [{var_name: row[var_name]['value']
                              if var_name in row else None
                   for var_name in var_names}
                  for row in raw_res['results']['bindings']]
        #var_names = [var_name for var_name in var_names]
        #return [var_names, values]
        return values

    def parse_result(self, res):
        raw_res = res
        common_res = res
        return common_res, raw_res

    def raw_query(self, qstr):
        return self.query(qstr)

    def query(self, qstr, is_update=False):
        sparql = self._get_sparql()
        if is_update:
            sparql.setMethod(POST)
        else:
            sparql.setMethod(GET)
        sparql.setReturnFormat(JSON)
        qstr = sparql_prefix + qstr
        sparql.setHTTPAuth
        sparql.setQuery(qstr)
        raw_res = sparql.query().convert()
        if sparql.queryType == SELECT:
            res = self._format_select_res(raw_res)
        elif sparql.queryType == INSERT:
            res = raw_res # TODO: Error handling here
        elif sparql.queryType == 'LOAD':
            res = raw_res # TODO: Error handling here
        else:
            res = raw_res
        return res

    def _create_insert_query(self, triples):
        q = """
            INSERT DATA {{
                GRAPH <{0}> {{
            """.format(self.base_graph)
        for triple in triples:
#            triple_str = ' '.join([term.n3() for term in triple]) + ' .\n'
            triple_str = ' '.join(['<{0}>'.format(str(term)) for term in triple]) + ' .\n'
            q += triple_str
        q += """}
            }
            """
        return q

    def _is_bool(self, s):
        s = s.lower()
        if s == 'true' or s == 'false':
            return True
        else:
            return False

    def _str2bool(self, s):
        s = s.lower()
        if s == 'true':
            return True
        elif s == 'false':
            return False
        else:
            raise Exception('{0} is not convertible to boolean'.format(s))

    def _is_float(self, s):
        try:
            float(s)
            return True
        except:
            return False

    def _parse_term(self, term):
        if isinstance(term, URIRef) or isinstance(term, Literal):
            return term
        elif isinstance(term, str):
            if 'http' == term[0:4]:
                node = URIRef(term)
            elif ':' in term: #TODO: This condition is dangerous.
                [ns, id_] = term.split(':')
                ns = self.namespaces[ns]
                node = ns[id_]
            else:
                if term.isdigit():
                    term = int(term)
                elif self._is_float(term):
                    term = float(term)
                if self._is_bool(term):
                    term = _str2bool(term)
                node = Literal(term)
        else:
            node = Literal(term)
        return node

    def add_triple(self, pseudo_s, pseudo_p, pseudo_o):
        triple = self.make_triple(pseudo_s, pseudo_p, pseudo_o)
        self.add_triples([triple])

    def add(self, triple):
        self.add_triples([triple])

    def make_triple(self, pseudo_s, pseudo_p, pseudo_o):
        s = self._parse_term(pseudo_s)
        p = self._parse_term(pseudo_p)
        o = self._parse_term(pseudo_o)
        return (s, p, o)

    def add_triples(self, pseudo_triples):
        triples = [self.make_triple(*pseudo_triple)
                   for pseudo_triple in pseudo_triples]
        self._add_triples(triples)

    def _add_triples(self, triples):
        q = self._create_insert_query(triples)
        res = self.update(q)

    def add_brick_instance(self, entity_name, tagset):
        entity = URIRef(BASE + entity_name)
        tagset = URIRef(BRICK + tagset)
        triples = [(entity, RDF.type, tagset)]
        self._add_triples(triples)
        return str(entity)

    def load_ttlfile(self, filepath):
        q = """
        load <file://{0}> into <{1}>
        """.format(filepath, self.base_graph)
        res = self.update(q)

    def load_schema(self):
        self.load_ttlfile(BRICK_FILE)
        self.load_ttlfile(BF_FILE)

    def parse(self, filepath, format=None):
        self.load_ttlfile(filepath)

    def serialize(self):
        qstr = """
        select ?s ?p ?o where{
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?s), "%s"))
        }
        """ % (BASE)
        res = self.raw_query(qstr)
        return res

    def __add__(self, other):
        assert isinstance(other, BrickEndpoint)
        qstr = """
        select ?s ?p ?o where{
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?s), "%s"))
        }
        """ % (BASE)
        res = other.raw_query(qstr)
        triples = [(URIRef(row['s']), URIRef(row['p']), URIRef(row['o'])) for row in res]
        triple_chunks = chunks(triples, 300)
        for chunk in triple_chunks:
            self._add_triples(chunk)
        return self



if __name__ == '__main__':
    endpoint = BrickEndpoint('http://localhost:8890/sparql', '1.0.3')
    endpoint.load_schema()
    test_qstr = """
        select ?s where {
        ?s rdfs:subClassOf+ brick:Temperature_Sensor .
        }
        """
    res = endpoint.query(test_qstr)
    print(res)
