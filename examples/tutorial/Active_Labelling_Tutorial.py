import sys
import json
import pdb
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count, combinations
import pickle
import scipy
from scipy.special import softmax
import sys
from sklearn.feature_extraction.text import CountVectorizer as CV
import re
import copy
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn_crfsuite import CRF
import nltk
import argparse

import pandas as pd
import numpy as np

from plastering.metadata_interface import *
from plastering.common import *
from plastering.rdf_wrapper import get_top_class

from plastering.inferencers.zodiac import ZodiacInterface
from plastering.inferencers.apl import ActivePartialLabelling
from plastering.metadata_interface import LabeledMetadata, RawMetadata
from plastering.uis.cmdline_ui import ReplUi

target_building = 'bldg'
config = {
    'brick_version': '1.0.3',
    'brick_file': 'https://brickschema.org/schema/1.0.3/Brick.ttl',
    'brickframe_file': 'https://brickschema.org/schema/1.0.3/BrickFrame.ttl',
}

# Select labeled srcids (Not all the data are labeled yet.)
target_srcids = [doc.srcid for doc in LabeledMetadata.objects(building=target_building)]
print('target #: {0}'.format(len(target_srcids)))

ActiveLabelling = ActivePartialLabelling(target_building=target_building,
                         target_srcids=target_srcids,
                         config=config,
                         logging_configfile='',
                         metadata_types=['VendorGivenName',
                                         'BACnetName',
                                         'BACnetDescription',
                                         'BACnetUnit',
                                         ],
                         )

#all_tagsets = [tagset.lower().split('#')[-1] for tagset in zodiac.schema_g.get_all_tagsets()]
#zodiac.ui = ReplUi(all_tagsets, zodiac.pgid)


# You can run it to the end. It will automatically select examples and update the model until it reaches a certain confidence level.
ActiveLabelling.learn_auto()

print(ActiveLabelling.predict_proba(target_srcids, output_format="ttl"))

ActiveLabelling.save_model('test_model2.obj')

print("Saved")
load_crf = ActiveLabelling.load_model('test_model2.obj')
print("Model Loaded:")
ActiveLabelling.crf = load_crf
ActiveLabelling.learn_auto()
