import pdb
from uuid import uuid4 as gen_uuid

from .. import Inferencer
from ...rdflib_wrapper import *
from jasonhelper import bidict

class DummyQuiver(Inferencer):

    def __init__(self,
                 target_building,
                 target_srcids,
                 source_buildings=[],
                 ui=None,
                 config={}):
        if 'ground_truth_ttl' not in config:
            raise Exception('True Turtle file should be given for DummyQuiver')
        self.true_g = init_graph()
        self.true_g.parse(config['ground_truth_ttl'], format='turtle')
        super(DummyQuiver, self).__init__(
            target_building=target_building,
            target_srcids=target_srcids,
            ui=ui,
            config=config,
            framework_name='quiver')
        self.cache_vav_dict(self.true_g)

    def cache_vav_dict(self, true_g):
        qstr = """
        select ?point ?srcid ?vav where {
        ?vav a brick:vav.
        ?point bf:isPointOf ?vav .
        ?point bf:srcid ?srcid.
        }
        """
        res = query_sparql(true_g, qstr)
        self.point_vav_dict = bidict()
        for row in res:
            vav = row['vav']
            point_srcid = row['srcid']
            point = create_uri(point_srcid)
            self.point_vav_dict[point] = vav

    def predict_cached(self, target_srcids=[]):
        pred_g = init_graph(empty=True)
        #colocated_points = {}
        occs = self.get_occs()
        for occ in occs:
            if self.prior_g and self.prior_confidences\
                    [(occ, RDF.type, BRICK.occupied_command)] < 0.5:
                continue
            vav = self.point_vav_dict[occ]
            insert_triple(pred_g, (vav, RDF['type'], BRICK['vav']))
            points = self.point_vav_dict.inverse[vav]
            for point in points:
                insert_triple(pred_g, (point, BF['isPointOf'], vav))
        self.pred_g = pred_g
        return self.pred_g

    def predict(self, target_srcids=[]):
        return self.predict_cached(target_srcids)

    def get_occs(self):
        if self.prior_g:
            qstr = """
            select ?occ where {
                ?occ a brick:occupied_command.
            }
            """
            res = query_sparql(self.prior_g, qstr)
            occs = [row['occ'] for row in res]
        else:
            qstr = """
            select ?srcid where {
                ?occ a brick:occupied_command.
                ?occ bf:srcid ?srcid.
            }
            """
            res = query_sparql(self.true_g, qstr)
            srcids = [row['srcid'] for row in res]
            occs = [create_uri(srcid) for srcid in srcids]
        return occs

    def predict_raw(self, target_srcids=[]):
        pred_g = init_graph(empty=True)
        if self.target_building == 'ebu3b':
            qstr = """
            select ?occ ?occ_srcid ?point ?point_srcid where {
                ?occ a brick:occupied_command.
                ?occ bf:srcid ?occ_srcid .
                ?occ bf:isPointOf ?something .
                ?point bf:isPointOf ?something .
                ?point bf:srcid ?point_srcid .
                ?occ bf:isPointOf ?something .
            }
            """
        else:
            raise Exception('qstr should be rewritten for {0}'
                            .format(self.target_building))
        # TODO: Add confidences (==1)
        res = query_sparql(self.true_g, qstr)
        vav_dict = {}
        #for
        #    random_obj = create_uri(str(gen_uuid())) # This would be a VAV.
        for row in res:
            occ_srcid = str(row['occ_srcid'])
            if occ_srcid not in vav_dict:
                vav_dict[occ_srcid] = create_uri(str(gen_uuid())) # This would be a VAV.
            vav = vav_dict[occ_srcid]
            occ = create_uri(occ_srcid)
            point = create_uri(str(row['point_srcid']))
            insert_triple(pred_g, (point, BF['isPointOf'], vav))
            insert_triple(pred_g, (occ, BF['isPointOf'], vav))
            insert_triple(pred_g, (vav, RDF['type'], BRICK['vav']))
        self.pred_g = pred_g
        return pred_g

    def predict_dep(self, target_srcids=[]):
        pred_g = init_graph(empty=True)
        occs = self.get_occs()
        for occ in occs:
            if self.target_building == 'ebu3b':
                srcid = occ.split('#')[-1]
                qstr = """
                select ?point where {{
                  ?occ bf:srcid "{0}" .
                  ?occ bf:isPointOf ?something .
                  ?point bf:isPointOf ?something .
                  ?point a/rdfs:subClassOf* brick:point .
                }}
                """.format(srcid)
            else:
                qstr = """
                select ?point where {{
                  {0} bf:isPointOf ?something .
                  ?point bf:isPointOf ?something .
                  ?point a/rdfs:subClassOf* brick:point .
                }}
                """.format(occ.n3())
            res = query_sparql(self.true_g, qstr)
            points = [row['point'] for row in res]
            random_obj = create_uri(str(gen_uuid())) # This would be a VAV.
            for point in points:
                insert_triple(pred_g, (point, BF['isPointOf'], random_obj))
                insert_triple(pred_g, (random_obj, RDF['type'], BRICK['VAV']))

        pred_g.serialize('test.ttl', format='turtle')
        self.pred_g = pred_g
        print('Quiver done')
        return pred_g

class DummyPritoni(Inferencer):

    def __init__(self,
                 ground_truth_ttl,
                 target_building,
                 target_srcids,
                 source_buildings=[],
                 ui=None,
                 config={}):
        self.true_g = init_graph()
        self.true_g.parse(ground_truth_ttl, format='turtle')
        super(DummyPritoni, self).__init__(
            target_building=target_building,
            target_srcids=target_srcids,
            ui=ui,
            config=config,
            framework_name='quiver')

    def get_ahu_datsp(self): #discharge air temperature setpoint
        qstr = """
        select ?datsp ?ahu where {
        ?datsp a/rdfs:subClassOf* brick:Discharge_Air_Temperature_Setpoint.
        ?datsp bf:isPointOf ?ahu.
        ?ahu a brick:AHU.
        }
        """
        res = query_sparql(self.prior_g, qstr)
        return [
            {
                'datsp': row['datsp'],
                'ahu': row['ahu']
            } for row in res
        ]

    def get_all_vavs_with_znt(self):
        qstr = """
        select ?vav where {
        ?vav a/rdfs:subClassOf* brick:VAV .
        ?znt bf:isPointOf ?vav.
        ?znt a brick:Zone_Temperature_Sensor.
        }
        """
        res = query_sparql(self.prior_g + self.schema_g, qstr)
        return [row['vav'] for row in res]

    def predict(self):
        pred_g = init_graph(True)
        ahu_datsps = self.get_ahu_datsp()
        found_vavs = self.get_all_vavs_with_znt()

        for row in ahu_datsps:
            ahu = row['ahu']
            datsp = row['datsp']
            qstr = """
            select ?vav where {{
              {0} bf:feeds+ ?vav.
              ?vav a/rdfs:subClassOf* brick:VAV.
            }}
            """.format(ahu.n3())
            res = query_sparql(self.true_g, qstr)
            true_vavs = [row['vav'] for row in res]
            pred_vavs = [vav for vav in true_vavs if vav in found_vavs]
            for vav in pred_vavs:
                insert_triple(pred_g, (ahu, BF['feeds'], vav))

        return pred_g

