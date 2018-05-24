import numpy as np
import re
import pdb

from collections import defaultdict as dd

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.preprocessing import LabelEncoder as LE

from . import Inferencer
from .algorithm.active_learning import active_learning
from ..metadata_interface import *


def get_name_features(names):

        name = []
        for i in names:
            s = re.findall('(?i)[a-z]{2,}',i)
            name.append(' '.join(s))

        cv = CV(analyzer='char_wb', ngram_range=(3,4))
        fn = cv.fit_transform(name).toarray()

        return fn


class ActiveLearningInterface(Inferencer):

    def __init__(self,
        target_building,
        target_srcids,
        fold,
        rounds
        ):

        super(ActiveLearningInterface, self).__init__(
            target_building=target_building,
            target_srcids=target_srcids
        )

        #Merged Initializations
        self.fold = fold
        self.rounds = rounds

        #TODO: pt_name is the raw vendorgiven name and pt_type is the corresponding tagset in brick volcabulary
        srcids = [point['srcid'] for point
                  in LabeledMetadata.objects(building=target_building)]
        pt_type = [LabeledMetadata.objects(srcid=srcid).first().point_tagset
                   for srcid in srcids]
        pt_name = [RawMetadata.objects(srcid=srcid).first()\
                   .metadata['VendorGivenName'] for srcid in srcids]
        self.fn = get_name_features(pt_name)
        le = LE()
        self.label = le.fit_transform(pt_type)

        self.learner = active_learning(
            self.fold,
            self.rounds,
            self.fn,
            self.label
        )


    def example_set():
        #TODO: get a set of example IDs that the user can provide label for, i.e, the set of examples to run AL
        pass


    def get_label(idx):
        #TODO: get the label for the example[idx] from human
        pass


    def select_example(self):

        idx, c_idx = self.learner.select_example()

        return idx


    def update_model(self, srcid, cluster_id):

        self.learner.labeled_set.append(srcid)
        self.learbner.new_ex_id = srcid
        self.learner.cluster_id = cluster_id
        self.learner.update_model()


    def predict(self, target_srcids):

        return self.learner.clf.predict(target_srcids)


    def run_auto(self):

        self.learner.run_CV()

