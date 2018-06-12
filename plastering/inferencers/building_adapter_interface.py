import numpy as np
import re
import time
import pdb
import pickle as pk

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.preprocessing import LabelEncoder as LE

from .algorithm.transfer_learning import transfer_learning
from . import Inferencer
from ..timeseries_interface import *
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


def get_data_features(building, start_time, end_time):

    res = read_from_db(building, start_time, end_time)

    X = []
    srcids = []
    ctr = 0
    ctr1 = 0
    for labeled in LabeledMetadata.objects(building=building):
        srcid = labeled.srcid
        ctr1 += 1
        try:
            data = res[srcid]
        except:
            print (srcid, 'not found and skipped.')
            continue

        #TODO: better handle the dimension, it's really ugly now
        #computing features on long sequence is really slow now, so only loading a small port of the readings
        try:
            df = data['data'][:3000]
            if len(df) < 400: #discard short sequences
                ctr += 1
                continue
            else:
                X.append( df )
        except:
            pdb.set_trace()

        srcids.append(srcid)

    print (ctr,'out of',ctr1,'points timeseries not loaded')

    min_len = min([len(x) for x in X])
    X = [x[:min_len] for x in X]
    dfe = data_feature_extractor( np.asarray(X) )
    fd = dfe.getF_2015_Hong()

    assert (len(srcids)==fd.shape[0])
    print ( 'data features for %s with dim:'%building, fd.shape)
    return srcids, fd


def get_namefeatures_labels(building):

    srcids = [point['srcid'] for point in LabeledMetadata.objects(building=building)]

    pt_type = [LabeledMetadata.objects(srcid=srcid).first().point_tagset.lower() for srcid in srcids]
    pt_name = [RawMetadata.objects(srcid=srcid).first().metadata['VendorGivenName'] for srcid in srcids]

    fn = get_name_features(pt_name)
    print ('%d point names loaded for %s'%(len(pt_name), building))

    return { srcid:[name_feature, label] for srcid,name_feature,label in zip(srcids,fn,pt_type) }


class BuildingAdapterInterface(Inferencer):

    def __init__(self,
                 target_building,
                 target_srcids,
                 source_buildings,
                 config={},
                 load_from_file=0
                 ):
        super(BuildingAdapterInterface, self).__init__(
            target_building=target_building,
            source_buildings=source_buildings,
            target_srcids=target_srcids
        )

        #gather the source/target data and name features, labels
        #TODO: handle multiple source buildings

        if 'source_time_ranges' in config:
            self.source_time_ranges = config['source_time_ranges']
            assert len(self.source_time_ranges) == len(source_buildings)
        else:
            self.source_time_ranges = [(None, None)]\
                * len(source_buildings)
        if 'target_time_range' in config:
            self.target_time_range = config['target_time_range']
        else:
            self.target_time_range = (None, None)

        if 'threshold' in config:
            self.threshold = config['threshold']
        else:
            self.threshold = 0.5

        source_building = source_buildings[0]

        if not load_from_file:
            #data features
            source_ids, train_fd = get_data_features(source_building,
                                                     self.source_time_ranges[0][0],
                                                     self.source_time_ranges[0][1])
            target_ids, test_fd = get_data_features(target_building,
                                                    self.target_time_range[0],
                                                    self.target_time_range[1])

            #name features, labels
            source_res = get_namefeatures_labels(source_building)
            train_label = [source_res[srcid][1] for srcid in source_ids]

            target_res = get_namefeatures_labels(target_building)
            test_fn = np.asarray( [target_res[tgtid][0] for tgtid in target_ids] )
            test_label = [target_res[tgtid][1] for tgtid in target_ids]

            #find the label intersection
            intersect = list( set(test_label) & set(train_label) )
            print ('intersected tagsets:', intersect)

            #preserve the intersection, get ids for indexing data feature matrices
            if intersect:
                train_filtered = [[i,j] for i,j in enumerate(train_label) if j in intersect]
                train_id, train_label = [list(x) for x in zip(*train_filtered)]
                test_filtered = [[i,j] for i,j in enumerate(test_label) if j in intersect]
                test_id, test_label = [list(x) for x in zip(*test_filtered)]
            else:
                raise ValueError('no common labels!')

            train_fd = train_fd[train_id, :]
            test_fd = test_fd[test_id, :]
            #test_fn = [test_fn[tid] for tid in test_id]
            test_fn = test_fn[test_id, :]
            print ('%d training examples left'%len(train_fd))
            print ('%d testing examples left'%len(test_fd))

            le = LE()
            le.fit(intersect)
            train_label = le.transform(train_label)
            test_label = le.transform(test_label)

            res = [train_fd, test_fd, train_label, test_label, test_fn]
            with open('./%s-%s.pkl'%(source_building,target_building), 'wb') as wf:
                pk.dump(res, wf)
        else:
            with open('./%s-%s.pkl'%(source_building,target_building), 'rb') as rf:
                res = pk.load(rf)
            train_fd, test_fd, train_label, test_label, test_fn = \
            res[0], res[1], res[2], res[3], res[4]

        print ( '# of classes:', len(set(train_label)) )
        print ( 'data features for %s with dim:'%source_building, train_fd.shape)
        print ( 'data features for %s with dim:'%target_building, test_fd.shape)


        self.learner = transfer_learning(
            train_fd,
            test_fd,
            train_label,
            test_label,
            test_fn,
            threshold = self.threshold
        )


    def predict(self):

        preds, labeled_set = self.learner.predict()

        return preds, labeled_set


    def run_auto(self):

        self.learner.run_auto()

