import json
from collections import OrderedDict
import random
import pdb
from copy import copy, deepcopy
from functools import reduce

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import vstack
import numpy as np

from brick_parser import tagsetList as tagset_list,\
                         tagList as tag_list
from building_tokenizer import get_unit_dict, get_bacnettype_dict
from mongo_models import *


tagset_list = list(set(tagset_list))


# Note:
# May need to remove "unknown" data points from learning samples?
def leave_one_word(s, w):
    if w in s:
        s = s.replace(w, '')
        s = w + '-' + s
    return s


def _bilou_tagset_phraser(sentence, token_labels):
    phrase_labels = list()
    curr_phrase = ''
    for i, (c, label) in enumerate(zip(sentence, token_labels)):
        if label[2:] in ['right_identifier', 'left_identifier']:
            continue
        tag = label[0]
        if tag=='B':
            if curr_phrase:
            # Below is redundant if the other tags handles correctly.       
                phrase_labels.append(curr_phrase)
            curr_phrase = label[2:]
        elif tag == 'I':
            if curr_phrase != label[2:]:
                phrase_labels.append(curr_phrase)
                curr_phrase = label[2:]
        elif tag == 'L':
            if curr_phrase != label[2:]:
                # Add if the previous label is different                    
                phrase_labels.append(curr_phrase)
            # Add current label                                             
            phrase_labels.append(label[2:])
            curr_phrase = ''
        elif tag == 'O':
            # Do nothing other than pushing the previous label              
            if curr_phrase:
                phrase_labels.append(curr_phrase)
            curr_phrase = ''
        elif tag == 'U':
            if curr_phrase:
                phrase_labels.append(curr_phrase)
            phrase_labels.append(label[2:])
        else:
            print('Tag is incorrect in: {0}.'.format(label))
            try:
                assert(False)
            except:
                pdb.set_trace()
    if curr_phrase != '':
        phrase_labels.append(curr_phrase)
    phrase_labels = [leave_one_word(\
                         leave_one_word(phrase_label, 'left_identifier'),\
                            'right_identifier')\
                        for phrase_label in phrase_labels]
    adder = lambda x, y: x + y
    splitter = lambda x: x.split('_')
    phrase_labels = list(reduce(adder, map(splitter, phrase_labels), []))
    return phrase_labels

def splitter(s):
    return s.split('_')


def translate_tagset_vector(tagset_vec, tagset_list=tagset_list):
    return [tagset for tagset, v in zip(tagset_list, tagset_vec) if v == 1]


def test_brick_tagset(sentence, token_labels, classifier, vectorizer):
    phrases = _bilou_tagset_phraser(sentence, token_labels)
    vec = vectorizer.transform([' '.join(phrases)])
    pred = classifier.predict(vec)[0]
    return translate_tagset_vector(pred)

def make_phrase_dict(sentence_dict, token_label_dict):
    phrase_dict = OrderedDict()
    for srcid, sentence in sentence_dict.items():
        token_labels = token_label_dict[srcid]
        phrase_dict[srcid] = _bilou_tagset_phraser(sentence, token_labels)
    return phrase_dict


def batch_test_brick_tagset(sentence_dict, \
                            token_label_dict, \
                            classifier, \
                            vectorizer,
                            tagset_list=tagset_list,
                            binerizer=None):
    srcid_list = list(sentence_dict.keys())
    phrase_dict = make_phrase_dict(sentence_dict, token_label_dict)
    doc = [' '.join(phrase_dict[srcid]) for srcid in srcid_list]
    vect_doc = vectorizer.transform(doc)

    certainty_dict = dict()
    pred_tagsets_dict = dict()
    pred_mat = classifier.predict(vect_doc)
    #prob_mat =classifier.predict_proba(vect_doc)
    for i, (srcid, pred) in enumerate(zip(srcid_list, pred_mat)):
        pred_tagsets_dict[srcid] = binerizer.inverse_transform(\
                                        np.asarray([pred]))[0]
        #pred_tagsets_dict[srcid] = translate_tagset_vector(pred, tagset_list)
        # TODO: Don't remove below. Activate this when using RandomForest
        #pred_vec = [prob[i][0] for prob in prob_mat]
        #certainty_dict[srcid] = sum(pred_vec) / float(len(pred)-sum(pred))
        certainty_dict[srcid] = 0

    return pred_tagsets_dict, certainty_dict


def learn_brick_tagsets(sentence_dict,
                        token_label_dict,
                        truth_dict,
                        classifier=RandomForestClassifier(),
                        vectorizer=TfidfVectorizer(ngram_range=(1, 2))):

    """
    Receive sentence_dict and label_dict at token-level (characeter-level)
    turth_dict is sentence-level ground truth.
    """

    phrase_dict = OrderedDict()
    for srcid, sentence in sentence_dict.items():
        phrase_dict[srcid] = _bilou_tagset_phraser(sentence, \
                                                   token_label_dict[srcid])
    with open('temp/phrases_ap_m.json', 'w') as fp:
        json.dump(phrase_dict, fp, indent=2)

    srcid_list = list(token_label_dict.keys())
    doc = [' '.join(phrase_dict[srcid]) for srcid in srcid_list]
    vect_doc = vectorizer.fit_transform(doc)
    assert len(srcid_list) == len(vect_doc.toarray())

    # Add tagsets (names) not defined in Brick
    undefined_tagsets = set()
    for srcid in sentence_dict.keys():
        truths = truth_dict[srcid]
        for truth in truths:
            if truth not in tagset_list:
                undefined_tagsets.add(truth)
    print('Undefined tagsets: {0}'.format(undefined_tagsets))
    tagset_list.extend(list(undefined_tagsets))
    truth_mat = list()
    feature_mat = list()
    for srcid in srcid_list:
        truths = truth_dict[srcid]
        truth_vector = [1 if tagset in truths else 0 for tagset in tagset_list]
        truth_mat.append(truth_vector)
        feature_mat.append(vect_doc[srcid_list.index(srcid)])
    feature_mat = vstack(feature_mat)
    #pdb.set_trace()

    classifier.fit(feature_mat, np.asarray(truth_mat))

    return classifier, vectorizer
