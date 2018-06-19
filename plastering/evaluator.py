from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pdb

def binarize_labels(true_labels, pred_labels):
    srcids = list(pred_labels.keys())
    tot_labels = [list(labels) for labels in
               list(pred_labels.values()) + list(true_labels.values())]
    mlb = MultiLabelBinarizer().fit(tot_labels)
    pred_mat = mlb.transform(pred_labels.values())
    true_mat = mlb.transform(true_labels.values())
    return true_mat, pred_mat

def get_micro_f1(true_labels, pred_labels):
    pred_mat, true_mat = binarize_labels(true_labels, pred_labels)
    return get_micro_f1_mat(true_mat, pred_mat)

def get_macro_f1(true_labels, pred_labels):
    pred_mat, true_mat = binarize_labels(true_labels, pred_labels)
    return get_macro_f1_mat(true_mat, pred_mat)

def get_macro_f1_mat(true_mat, pred_mat):
    assert true_mat.shape == pred_mat.shape
    f1s = []
    for i in range(0, true_mat.shape[1]):
        if 1 not in true_mat[:,i]:
            continue
        f1 = f1_score(true_mat[:,i], pred_mat[:,i])
        f1s.append(f1)
    return np.mean(f1s)

def get_multiclass_micro_f1(true_labels, pred_labels):
    le = LabelEncoder()
    #pred_mat, true_mat = binarize_labels(true_labels, pred_labels)
    #f1_custom = get_micro_f1_mat(true_mat, pred_mat)
    srcids = list(true_labels.keys())
    true_label_list = [true_labels[srcid] for srcid in srcids]
    pred_label_list = [pred_labels[srcid] for srcid in srcids]
    le = LabelEncoder()
    le.fit(true_label_list + pred_label_list)
    true_encoded = le.transform(true_label_list)
    pred_encoded = le.transform(pred_label_list)
    f1_micro = f1_score(true_encoded, pred_encoded, average='micro')
    #f1_weighted = f1_score(true_encoded, pred_encoded, average='weighted')
    #pdb.set_trace()
    return f1_micro

def get_multiclass_macro_f1(true_labels, pred_labels):
    le = LabelEncoder()
    #pred_mat, true_mat = binarize_labels(true_labels, pred_labels)
    #f1_custom = get_micro_f1_mat(true_mat, pred_mat)
    srcids = list(true_labels.keys())
    true_label_list = [true_labels[srcid] for srcid in srcids]
    pred_label_list = [pred_labels[srcid] for srcid in srcids]
    le = LabelEncoder()
    le.fit(true_label_list + pred_label_list)
    true_encoded = le.transform(true_label_list)
    pred_encoded = le.transform(pred_label_list)
    f1_micro = f1_score(true_encoded, pred_encoded, average='macro')
    #f1_weighted = f1_score(true_encoded, pred_encoded, average='weighted')
    #pdb.set_trace()
    return f1_micro



def get_micro_f1_mat(true_mat, pred_mat):
    TP = np.sum(np.bitwise_and(true_mat==1, pred_mat==1))
    TN = np.sum(np.bitwise_and(true_mat==0, pred_mat==0))
    FN = np.sum(np.bitwise_and(true_mat==1, pred_mat==0))
    FP = np.sum(np.bitwise_and(true_mat==0, pred_mat==1))
    micro_prec = TP / (TP + FP)
    micro_rec = TP / (TP + FN)
    return 2 * micro_prec * micro_rec / (micro_prec + micro_rec)

def get_point_accuracy(true_tagsets, pred_tagsets):
    target_srcids = pred_tagsets.keys()
    return sum([true_tagsets[srcid].lower() == pred_tagsets[srcid].lower()
                for srcid in target_srcids]) / len(target_srcids)

def get_accuracy(true_tagsets, pred_tagsets):
    pass #TODO

def get_set_accuracy(true_label_sets, pred_tagset_sets):
    # Accuracy per sample = #intersection / #union
    # Accuracy over set = average of the accuracy per sample
    # Input params dictionary based on the srcids
    for srcid, pred_tagset_set in pred_tagset_sets.items():
        pass #TODO

