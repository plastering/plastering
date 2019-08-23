import json
import pdb
import logging

from plastering.inferencers.zodiac import ZodiacInterface
from plastering.metadata_interface import LabeledMetadata, RawMetadata
from plastering.uis.cmdline_ui import ReplUi

target_building = 'ap_m'
config = {
    'brick_version': '1.0.3',
    'brick_file': '/home/jbkoh/repo/Brick/dist/Brick.ttl',
    'brickframe_file': '/home/jbkoh/repo/Brick/dist/BrickFrame.ttl',
}
#pgid = 'master'
#pgid = '1f520732-240f-4d12-9b05-0a572e2e90f1'

# Select labeled srcids (Not all the data are labeled yet.)
labeled_list = LabeledMetadata.objects(building=target_building)
#target_srcids = [labeled.srcid for labeled in RawMetadata.objects(building=target_building)]
target_srcids = [labeled['srcid'] for labeled in labeled_list]

zodiac = ZodiacInterface(target_building=target_building,
                         target_srcids=target_srcids,
                         config=config,
                         logging_configfile='config/logging.yaml',
                         metadata_types=['VendorGivenName'],
                         )
all_tagsets = [tagset.lower().split('#')[-1] for tagset in zodiac.schema_g.get_all_tagsets()]
zodiac.ui = ReplUi(all_tagsets, zodiac.pgid)
#tolearn_srcids = [doc.srcid for doc in LabeledMetadata.objects(pgid='1f520732-240f-4d12-9b05-0a572e2e90f1') if doc.srcid in target_srcids]
#pdb.set_trace()
#zodiac.update_model(tolearn_srcids)
zodiac.learn_auto()
pred_g, proba = zodiac.predict_proba(target_srcids, output_format='ttl')
pred_g.g.serialize('result/zodiac_inferred_graph.jsonld', format='json-ld')
pred_g.g.serialize('result/zodiac_inferred_graph.ttl', format='turtle')
