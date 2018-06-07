import numpy as np
import re
import time

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.preprocessing import LabelEncoder as LE

from .algorithm.transfer_learning import transfer_learning
from . import Inferencer
from ..timeseries_inferface import *
from ..metadata_interface import *
from ..data_feature_extractor import *


def get_name_features(names):

    name = []
    for i in names:

        s = re.findall('(?i)[a-z]{2,}',i)
        name.append(' '.join(s).lower())

    cv = CV(analyzer='char_wb', ngram_range=(3,4))
    fn = cv.fit_transform(name).toarray()

    return fn


def get_data_features(building):

    res = read_from_db(building)

    fd = []
    for point, data in res.items():
        #t0 = time.clock()
        #TODO: better handle the dimension, it's really ugly now
        dfe = data_feature_extractor( data['data'][:2000].as_matrix().reshape(1,-1) )
        fd.append( dfe.getF_2015_Hong().ravel() )
        #print (time.clock() - t0)

    print ( 'data features for %s with dim:'%building, np.asarray(fd).shape)
    return np.asarray(fd)


def get_namefeatures_labels(building):

    srcids = [point['srcid'] for point in LabeledMetadata.objects(building=building)]
    pt_type = [LabeledMetadata.objects(srcid=srcid).first().point_tagset.lower() for srcid in srcids]
    pt_name = [RawMetadata.objects(srcid=srcid).first().metadata['VendorGivenName'] for srcid in srcids]
    fn = get_name_features(pt_name)
    print ('%d point names loaded for %s'%(len(pt_name), building))

    return fn, pt_type


class BuildingAdapterInterface(Inferencer):

    def __init__(self,
        target_building,
        target_srcids,
        source_buildings,
        ):

        super(BuildingAdapterInterface, self).__init__(
            target_building=target_building,
            source_buildings=[src for src in source_buildings],
            target_srcids=target_srcids
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

        #data features
        train_fd = get_data_features('ucsd')
        test_fd = get_data_features('rice')

        #labels, name features for tgt_bldg
        test_fn, test_label = get_namefeatures_labels('uva_cse')
        _, train_label = get_namefeatures_labels('ap_m')

        #find the class intersection
        intersect = set(test_label) & set(train_label)
        print (intersect)

        #preserve only the intersected, id used for indexing data feature matrices
        if intersect:
            train_filtered = [[i,j] for i,j in enumerate(train_label) if j in intersect]
            train_id, train_label = [list(x) for x in zip(*train_filtered)]
            test_filtered = [[i,j] for i,j in enumerate(test_label) if j in intersect]
            test_id, test_label = [list(x) for x in zip(*test_filtered)]
        else:
            raise ValueError('no common labels!')

        train_fd = train_fd[train_id, :]
        test_fd = test_fd[test_id, :]
        test_fn = test_fn[test_id, :]

        le = LE()
        le.fit(intersect)
        train_label = le.transform(train_label)
        test_label = le.transform(test_label)


        self.learner = transfer_learning(
            train_fd,
            test_fd,
            train_label,
            test_label,
            test_fn,
        )


    def predict(self):

        preds, labeled_set = self.learner.predict()

        return preds, labeled_set


    def run_auto(self):

        self.learner.run_auto()

