from .. import Inferencer
from ...rdflib_wrapper import *

from uuid import uuid4 as gen_uuid

class DummyQuiver(Inferencer):

    def __init__(self,
                 ground_truth_ttl,
                 target_building,
                 target_srcids,
                 source_buildings=[],
                 ui=None,
                 config={}):
        self.true_g = init_graph()
        self.true_g.parse(ground_truth_ttl, format='turtle')
        super(GroundTruthInterface, self).__init__(
            target_building=target_building,
            target_srcids=target_srcids,
            ui=ui,
            config=config,
            framework_name='quiver')

    def get_occs(self):
        qstr = """
        select ?occ where {
        ?occ a/rdfs:subClassOf* brick:Occupied_Command.
        }
        """
        res = query_sparql(self.prior_g, qstr)
        return [row['occ'] for row in res]


    def predict(self):
        pred_g = init_graph()
        occs = self.get_occs()
        for occ in occs:
            qstr = """
            select ?point where {{
              {0} bf:isPointOf ?something .
              ?point bf:isPointOf ?something .
              ?point a/rdfs:subClassOf* brick:Point .
            }}
            """.format(occ.n3())
            res = query_sparql(self.true_g, qstr)
            points = [row['point'] for row in res]
            random_obj = create_uri(str(gen_uuid())) # This would be a VAV.
            for point in points:
                insert_triple(pred_g, (point, BF['isPointOf'], random_obj))
                insert_triple(pred_g, (random_obj, RDF['type'], BRICK['VAV']))

        pred_g.serialize('test.ttl', format='turtle')
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
        res = query_sparql(self.prior_g, qstr)
        return [row['vav'] for row in res]

    def predict(self):
        pred_g = init_graph()
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
            pdb.set_trace()
            res = query_sparql(self.true_g, qstr)
            true_vavs = [row['vav'] for row in res]
            pred_vavs = [vav for vav in true_vavs if vav in found_vavs]
            for vav in pred_vavs:
                insert_triple(pred_g, (ahu, BF['feeds'], vav))

        pred_g.serialize('test.ttl', format='turtle')
        return pred_g










