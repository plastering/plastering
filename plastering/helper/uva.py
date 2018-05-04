import pandas as pd

from ..metadata_interface import *

def load_uva_building(building='uva_cse',
                      filename='./groundtruth/uva_cse_point_map.csv'):
    df = pd.read_csv(filename)
    for i, row in df.iterrows():
        srcid = row['original label']
        tagset = row['tagset']

        # Store raw metadata
        raw_obj = RawMetadata.objects(srcid=srcid)\
            .upsert_one(srcid=srcid, building=building)
        raw_obj.metadata['VendorGivenName'] = srcid
        raw_obj.save()

        # Store ground truth
        labeled_obj = LabeledMetadata.objects(srcid=srcid)\
            .upsert_one(srcid=srcid, building=building)
        labeled_obj.point_tagset = tagset
        labeled_obj.save()

