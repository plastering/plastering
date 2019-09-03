import json
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

# Select labeled srcids (Not all the data are labeled yet.)
target_srcids = [doc.srcid for doc in LabeledMetadata.objects(building=target_building)]
print('target #: {0}'.format(len(target_srcids)))
zodiac = ZodiacInterface(target_building=target_building,
                         target_srcids=target_srcids,
                         config=config,
                         logging_configfile='config/logging.yaml',
                         metadata_types=['VendorGivenName',
                                         'BACnetName',
                                         'BACnetDescription',
                                         'BACnetUnit',
                                         ],
                         )
all_tagsets = [tagset.lower().split('#')[-1] for tagset in zodiac.schema_g.get_all_tagsets()]
zodiac.ui = ReplUi(all_tagsets, zodiac.pgid)


# You can run it to the end. It will automatically select examples and update the model until it reaches a certain confidence level.
zodiac.learn_auto()


# Otherwise, you can do it manually, step by step.
# selected_samples = zodiac.select_informative_samples(5)
# zodiac.update_model(selected_samples)

pred_g, proba = zodiac.predict_proba(target_srcids, output_format='ttl')
pred_g.g.serialize('result/zodiac_inferred_graph.jsonld', format='json-ld')
pred_g.g.serialize('result/zodiac_inferred_graph.ttl', format='turtle')
