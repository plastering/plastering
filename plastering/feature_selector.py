import re
import pickle as pk
import numpy as np

from .timeseries_interface import *
from .metadata_interface import *
from .data_feature_extractor import *
from .inferencers.building_adapter_interface import *

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.preprocessing import LabelEncoder as LE

from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectFromModel as SFM
from sklearn.metrics import accuracy_score as ACC


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

        try:
            df = data['data'][:3000]
            if len(df) < 400: #discard short sequences
                ctr += 1
                continue
            else:
                X.append( df )
        except:
            pass

        srcids.append(srcid)

    print (ctr, 'out of', ctr1, 'points timeseries not loaded')

    min_len = min([len(x) for x in X])
    X = [x[:min_len] for x in X]
    dfe = data_feature_extractor( np.asarray(X) )

    #res = list(map(lambda x: eval('dfe.' + x + '()'), dfe.functions))
    funcs = [getattr(dfe, func) for func in dfe.functions]
    res = [func() for func in funcs]
    fd = np.concatenate(res, axis=1)
    #fd = dfe.getF_2015_Hong()

    assert (len(srcids)==fd.shape[0])
    print ( 'data features for %s with dim:'%building, fd.shape)

    return srcids, fd


def get_labels(building):

    srcids = [point['srcid'] for point in LabeledMetadata.objects(building=building)]

    pt_type = [LabeledMetadata.objects(srcid=srcid).first().point_tagset.lower() for srcid in srcids]

    print ('%d point names loaded for %s'%(len(pt_type), building))

    return { srcid: label for srcid,label in zip(srcids,pt_type) }


def get_CV_acc(X, Y, clf):

    kf = KFold(n_splits=10)
    acc = []
    for train, test in kf.split(X):

        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]

        clf.fit(X_train, Y_train)
        acc.append( ACC( clf.predict(X_test), Y_test ) )

    return np.mean(acc)


class feature_selector():

    def __init__(self, target_building, method, load_from_file=1):

        self.time_range = (None, None)

        if not load_from_file:
            #data features
            ids, self.fd = get_data_features(target_building,
                                        self.time_range[0],
                                        self.time_range[1])
            print ('%d data streams loaded'%len(self.fd))

            #labels
            res = get_labels(target_building)
            label = [res[srcid] for srcid in ids]
            le = LE()
            self.label = le.fit_transform(label)

            res = [self.fd, self.label]
            with open('./%s_fs.pkl'%(target_building), 'wb') as wf:
                pk.dump(res, wf)
        else:
            with open('./%s_fs.pkl'%(target_building), 'rb') as rf:
                res = pk.load(rf)
            self.fd, self.label = res[0], res[1]

        print ( '# of classes:', len(set(self.label)) )
        print ( 'data features for %s with dim:'%target_building, self.fd.shape)

        self.method = method


    def run_auto(self):


        clf = LinearSVC(C=0.1, penalty="l1", dual=False)
        #rf = RFC(n_estimators=100, criterion='entropy')

        acc = get_CV_acc(self.fd, self.label, clf)
        print ('acc before selection is', acc)

        model = SFM(clf, prefit=True)
        fd_new = model.transform(self.fd)
        print ('dim after selection is', fd_new.shape)

        acc = get_CV_acc(fd_new, self.label, clf)
        print ('acc after selection is', acc)

        #rf.fit(self.train_fd, self.train_label)
        #pred = rf.predict(self.test_fd)
        #print ( 'direct data feature-based transfer acc on tgt_bldg:', ACC(pred, self.test_label) )

