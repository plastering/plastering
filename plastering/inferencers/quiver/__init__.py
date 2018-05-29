from .. import Inferencer
from ...rdflib_wrapper import *

from uuid import uuid4 as gen_uuid

class DummyQuiver(GroundTruthInterface):

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

        pred_g.serialize('test.ttl', format='turtle')
        return pred_g

class DummyPritoniEtal(Inferencer):
    pass
