import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer as CV

from .algorithm.transfer_learning import transfer_learning
from . import Inferencer
from ..timeseries_inferface import *
from ..data_feature_extrator import *


def get_name_features(names):

    name = []
    for i in names:

        s = re.findall('(?i)[a-z]{2,}',i)
        name.append(' '.join(s))

    cv = CV(analyzer='char_wb', ngram_range=(3,4))
    fn = cv.fit_transform(name).toarray()

    return fn


class BuildingAdapterInterface(Inferencer):

    def __init__(self,
        target_building,
        source_buildings,
        ):

        super(building_adapter_interface, self).__init__(
            target_building = target_building,
            source_buildings = [src for src in source_buildings]
        )

        #gather the training/testing data and name features
        '''
        #old block loading from pre-computed files
        input1 = np.genfromtxt('../data/rice_hour_sdh', delimiter=',')
        input2 = np.genfromtxt('../data/keti_hour_sum', delimiter=',')
        input3 = np.genfromtxt('../data/sdh_hour_rice', delimiter=',')
        input2 = np.vstack((input2, input3))
        fd1 = input1[:, 0:-1]
        fd2 = input2[:, 0:-1]

        train_fd = fd1
        test_fd = fd2
        train_label = input1[:, -1]
        test_label = input2[:, -1]

        pt_name = [i.strip().split('\\')[-1][:-5] for i in open('../data/rice_pt_sdh').readlines()]
        test_fn = get_name_features(pt_name)
        '''

        #tl = transfer_learning(train_fd, test_fd, train_label, test_label, test_fn, True)

        self.learner = transfer_learning(
            train_fd,
            test_fd,
            train_label,
            test_label,
            test_fn,
            switch=True
        )


    def predict(self):

        preds, labeled_set = self.learner.predict()

        return preds, labeled_set


    def run_auto(self):

        self.learner.run_auto()

