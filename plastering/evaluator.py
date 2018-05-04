

from copy import deepcopy
from sklearn.metrics import f1_score
import numpy as np


def get_macro_f1(true_mat, pred_mat):
    assert true_mat.shape == pred_mat.shape
    f1s = []
    for i in range(0, true_mat.shape[1]):
        if 1 not in true_mat[:,i]:
            continue
        f1 = f1_score(true_mat[:,i], pred_mat[:,i])
        f1s.append(f1)
    return np.mean(f1s)

def get_point_accuracy(true_tagsets, pred_tagsets):
    return sum([true == pred for true, pred 
                in zip(true_tagsets, pred_tagsets)]) / len(true_tagsets)
