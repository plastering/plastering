'''
buildsys15 - Building Adapter:
local weighted transfer learning btw buildings
'''
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as F1
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize
from collections import defaultdict as DD
from collections import Counter as CT
from matplotlib import cm as Color

import numpy as np
import re
import math
import random
import itertools
import pylab as pl
import matplotlib.pyplot as plt


def get_name_features(names):

    name = []
    for i in names:

        s = re.findall('(?i)[a-z]{2,}',i)
        name.append(' '.join(s))

    cv = CV(analyzer='char_wb', ngram_range=(3,4))
    fn = cv.fit_transform(name).toarray()

    return fn


def plot_confusion_matrix(test_label, pred):

    mapping = {1:'co2',2:'humidity',3:'pressure',4:'rmt',5:'status',6:'stpt',7:'flow',8:'HW sup',9:'HW ret',10:'CW sup',11:'CW ret',12:'SAT',13:'RAT',17:'MAT',18:'C enter',19:'C leave',21:'occu',30:'pos',31:'power',32:'ctrl',33:'fan spd',34:'timer'}
    cm_ = CM(test_label, pred)
    cm = normalize(cm_.astype(np.float), axis=1, norm='l1')
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=Color.YlOrBr)
    fig.colorbar(cax)
    for x in xrange(len(cm)):
        for y in xrange(len(cm)):
            ax.annotate(str("%.3f(%d)"%(cm[x][y], cm_[x][y])), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=9)
    cm_cls =np.unique(np.hstack((test_label, pred)))
    cls = []
    for c in cm_cls:
        cls.append(mapping[c])
    pl.yticks(range(len(cls)), cls)
    pl.ylabel('True label')
    pl.xticks(range(len(cls)), cls)
    pl.xlabel('Predicted label')
    pl.title('Confusion Matrix (%.3f)'%(ACC(pred, test_label)))
    pl.show()


def output_labels():
    #re-map class label to 0~N
    u, remap = np.unique(np.append(label,pred), return_inverse=True)
    remap = remap[-len(pred):]#output parameters for testing in Java
    f = open('TL_out','w')
    f.writelines(",".join(str(i) for i in l_id))
    f.write('\n')
    f.writelines(",".join(str(l) for l in remap))
    f.write('\n')
    f.close()


class transfer_learning:

    def __init__(self, train_fd, test_fd, train_label, test_label, test_fn, threshold=0.5, switch=False):

        self.train_fd = train_fd
        self.train_label = train_label

        self.test_fd = test_fd
        self.test_label = test_label

        self.test_fn = test_fn

        self.bl = []

        self.agreement_threshold = threshold
        if switch == True:

            fd_tmp = self.train_fd
            self.train_fd = self.test_fd
            self.test_fd = fd_tmp
            l_tmp = self.train_label
            self.train_label = self.test_label
            self.test_label = l_tmp

        assert self.test_fn.shape[0] == self.test_fd.shape[0]
        assert self.train_fd.shape[0] == len(self.train_label)
        assert self.test_fd.shape[0] == len(self.test_label)


    def get_base_learners(self):

        rf = RFC(n_estimators=100, criterion='entropy')
        svm = SVC(kernel='rbf', probability=True)
        lr = LR()
        self.bl = [rf, lr, svm] #set of base learners
        for b in self.bl:
            b.fit(self.train_fd, self.train_label) #train each base classifier


    def run_auto(self):

        '''
        test direct data feature based transfer accuracy on the new building
        '''
        rf = RFC(n_estimators=100, criterion='entropy')
        rf.fit(self.train_fd, self.train_label)
        pred = rf.predict(self.test_fd)
        print ( 'direct data feature-based transfer acc on tgt_bldg:', ACC(pred, self.test_label) )
        #plot_confusion_matrix(self.test_label, pred)


        '''
        step1: train base models from bldg1
        '''
        self.get_base_learners()


        '''
        step2: TL with name feature on bldg2
        '''
        label = self.test_label
        class_ = np.unique(self.train_label)

        for b in self.bl:
            print ( b.score(self.test_fd, label) )

        n_class = 32/2
        c = KMeans(init='k-means++', n_clusters=n_class, n_init=10)
        c.fit(self.test_fn)
        dist = np.sort(c.transform(self.test_fn))
        ex_id = DD(list) #example id for each C
        for i,j,k in zip(c.labels_, xrange(len(self.test_fn)), dist):
            ex_id[i].append(int(j))

        #getting neighors for each ex
        nb_c = DD() #nb from clustering results
        for exx in ex_id.values():
            exx = np.asarray(exx)
            for e in exx:
                nb_c[e] = exx[exx!=e]

        nb_f = [DD(), DD(), DD()] #nb from classification results
        for b,n in zip(self.bl, nb_f):
            preds = b.predict(self.test_fd)
            ex_ = DD(list)
            for i,j in zip(preds, xrange(len(self.test_fd))):
                ex_[i].append(int(j))
            for exx in ex_.values():
                exx = np.asarray(exx)
                for e in exx:
                    n[e] = exx[exx!=e]

        #use base learners' predicitons
        acc_ = []
        cov_ = []
        #for delta in np.linspace(0.1, 0.5, 5):
        for delta in np.linspace(self.agreement_threshold, self.agreement_threshold, 1):
            print ( 'running TL with agreement threshold =', delta )

            l_id = []
            output = DD()
            preds = np.array([999 for i in xrange(len(self.test_fd))])
            for i in xrange(len(self.test_fn)):
                #get the weight for each bl: by computing sim btw cluster and clf
                w = []
                v_c = set(nb_c[i])
                for n in nb_f:
                    v_f = set(n[i])
                    cns = len(v_c & v_f) / float(len(v_c | v_f)) #original count based weight
                    inter = v_c & v_f
                    union = v_c | v_f
                    d_i = 0
                    d_u = 0
                    for it in inter:
                        d_i += np.linalg.norm(self.test_fn[i]-self.test_fn[it])
                    for u in union:
                        d_u += np.linalg.norm(self.test_fn[i]-self.test_fn[u])
                    if len(inter) != 0:
                        sim = 1 - (d_i/d_u)/cns
                        #sim = (d_i/d_u)/cns

                    if i in output:
                        output[i].extend(['%s/%s'%(len(inter), len(union)), 1-sim])
                    else:
                        output[i] = ['%s/%s'%(len(inter), len(union)), 1-sim]
                    w.append(sim)
                output[i].append(np.mean(w))

                if np.mean(w) >= delta:
                    w[:] = [float(j)/sum(w) for j in w]
                    pred_pr = np.zeros(len(class_))
                    for wi, b in zip(w,self.bl):
                        pr = b.predict_proba(self.test_fd[i].reshape(1,-1))
                        pred_pr = pred_pr + wi*pr
                    preds[i] = class_[np.argmax(pred_pr)]
                    l_id.append(i)

            acc_.append( ACC(preds[preds!=999], label[preds!=999]) )
            cov_.append( 1.0 * len(preds[preds!=999])/len(label) )

        print ( 'acc =', acc_, ';' )
        print ( 'cov =', cov_, ';' )

        return preds[preds!=999], l_id


    def predict(self):

        preds, labeled_set = self.run_auto()
        assert len(preds) == len(labeled_set)

        return preds, labeled_set


if __name__ == "__main__":

    input1 = np.genfromtxt('../../data/rice_hour_sdh', delimiter=',')
    #input1 = np.genfromtxt('sdh_hour_soda', delimiter=',')
    #input1 = np.genfromtxt('soda_hour_rice', delimiter=',')
    input2 = np.genfromtxt('../../data/keti_hour_sum', delimiter=',')
    #input21 = np.genfromtxt('rice_hour_soda', delimiter=',')
    input3 = np.genfromtxt('../../data/sdh_hour_rice', delimiter=',')
    input2 = np.vstack((input2,input3))
    fd1 = input1[:, 0:-1]
    fd2 = input2[:, 0:-1]

    #self.train_fd = np.hstack((fd1,fd2))
    train_fd = fd1
    test_fd = fd2
    train_label = input1[:, -1]
    #self.test_label = np.hstack((input2[:,-1],input3[:,-1]))
    test_label = input2[:,-1]

    ptn = [i.strip().split('\\')[-1][:-5] for i in open('../../data/rice_pt_sdh').readlines()]
    test_fn = get_name_features(ptn)

    tl = transfer_learning(train_fd, test_fd, train_label, test_label, test_fn, switch=True)
    tl.run_auto()
    #preds, labeled = tl.predict()

