import json
from collections import defaultdict, OrderedDict
from copy import copy
import pdb

from pymongo import MongoClient
import arrow
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from mongo_models import summary_query_template

class Resulter():
    def __init__(self, token_type="BILOU", spec={}):
        self.db = MongoClient().get_database('scrabble')
        self.result_coll = self.db.get_collection('results')
        self.summary_coll = self.db.get_collection('summary')
        self.result_dict = defaultdict(dict)
        self.summary = dict()
        self.token_type = token_type
        self.spec = spec

    def calc_accuracy(true_tagsets_list, pred_tagsets_list):
        acc_list = list()
        for true_tagsets, pred_tagsets in zip(true_tagsets_list, \
                                                pred_tagsets_list):
            true_tagsets = set(true_tagsets)
            pred_tagsets = set(pred_tagsets)
            acc_list.append(len(true_tagsets.intersection(pred_tagsets))/\
                            len(true_tagsets.union(pred_tagsets)))
        return np.mean(acc_list)

    def add_results(self, srcids, sentences, pred_token_labels, orig_token_labels):
        for srcid in srcids:
            self.add_one_result(sentences[srcid],
                                pred_token_labels[srcid],
                                orig_token_labels[srcid])

    def add_one_result(self, srcid, sentence, \
                             pred_token_labels, orig_token_labels):
        # Check the size of data
        assert len(sentence) == len(pred_token_labels)
        assert len(orig_token_labels) == len(pred_token_labels)

        # Add raw results
        self.result_dict[srcid] = {
                'sentence': sentence,
                'pred_token_labels': pred_token_labels,
                'orig_token_labels': orig_token_labels
                }

        # Gen labels per phrase
        pred_phrase_labels = list()
        if self.token_type=='BILOU':
            phraser = self._bilou_phraser
        self.result_dict[srcid]['pred_phrase_labels'] \
                = phraser(pred_token_labels)
        self.result_dict[srcid]['orig_phrase_labels'] \
                = phraser(orig_token_labels)

    def measure_accuracy_by_phrase(self,
                                   pred_phrase_labels, 
                                   orig_phrase_labels,
                                   pessimistic_flag=False):
        #total_label_num = len(orig_phrase_labels)
        found_label_num = len(pred_phrase_labels)
        correct_label_num = 0
        total_label_num = 0
        for label in orig_phrase_labels:
            if pessimistic_flag and label.split('-')[0] in ['leftidentifier', 
                                             'rightidentifier', 
                                             'room',
                                             'building',
                                             'network_adapter']:
                # TODO: need to be fixed.
                if label in pred_phrase_labels:
                    pred_phrase_labels.remove(label)
                    found_label_num -= 1
                continue
            total_label_num += 1
            if label in pred_phrase_labels:
                correct_label_num += 1
                pred_phrase_labels.remove(label)
        incorrect_label_num = total_label_num - correct_label_num

        return found_label_num, correct_label_num, incorrect_label_num

    def measure_accuracy_by_token(self, pred_token_labels, orig_token_labels):
        assert(len(pred_token_labels)==len(orig_token_labels))
        correct_cnt = 0
        incorrect_cnt = 0
        for pred, orig in zip(pred_token_labels, orig_token_labels):
            if pred==orig:
                correct_cnt += 1
            else:
                incorrect_cnt += 1
        return correct_cnt, incorrect_cnt

    def summarize_result(self):

        #self.summary = {'specification': self.spec}
        # Calculate character level accuracy
        char_correct_cnt = 0
        char_total_cnt = 0

        correct_phrase_cnt = 0
        incorrect_phrase_cnt = 0
        predicted_phrase_cnt = 0

        pess_correct_phrase_cnt = 0
        pess_incorrect_phrase_cnt = 0
        pess_predicted_phrase_cnt = 0

        pred_tag = list()
        true_tag = list()
        pred_phrases = list()
        true_phrases = list()

        for srcid, result in self.result_dict.items():
            pred_tag += result['pred_token_labels']
            true_tag += result['orig_token_labels']
            pred_phrases.append(result['pred_phrase_labels'])
            true_phrases.append(result['orig_phrase_labels'])
            correct_cnt, incorrect_cnt = self.measure_accuracy_by_token(\
                                            result['pred_token_labels'],\
                                            result['orig_token_labels'])
            char_correct_cnt += correct_cnt
            char_total_cnt += (correct_cnt + incorrect_cnt)
            found, correct, incorrect  = self.measure_accuracy_by_phrase(
                                            copy(result['pred_phrase_labels']),
                                            copy(result['orig_phrase_labels']))
            correct_phrase_cnt += correct
            incorrect_phrase_cnt += incorrect
            predicted_phrase_cnt += found

            pess_found, pess_correct, pess_incorrect = \
                    self.measure_accuracy_by_phrase(
                            copy(result['pred_phrase_labels']),
                            copy(result['orig_phrase_labels']),
                            pessimistic_flag=True)
            pess_correct_phrase_cnt += pess_correct
            pess_incorrect_phrase_cnt += pess_incorrect
            pess_predicted_phrase_cnt += pess_found

        phrase_binarizer = MultiLabelBinarizer()
        phrase_binarizer.fit(pred_phrases + true_phrases)

        assert len(pred_tag) == len(true_tag)

        self.summary['char_precision'] = \
                float(char_correct_cnt)/char_total_cnt
        self.summary['phrase_precision'] = \
                float(correct_phrase_cnt) \
                / (correct_phrase_cnt + incorrect_phrase_cnt)
        self.summary['phrase_recall'] = \
                float(correct_phrase_cnt) / (predicted_phrase_cnt)
        self.summary['pessimistic_phrase_precision'] = \
                float(pess_correct_phrase_cnt) \
                / (pess_correct_phrase_cnt + pess_incorrect_phrase_cnt)
        self.summary['pessimistic_phrase_recall'] = \
                float(pess_correct_phrase_cnt) / (pess_predicted_phrase_cnt)
        _, _, char_macro_f1, _ = precision_recall_fscore_support(\
                                        true_tag, pred_tag, average='macro')
        self.summary['char_macro_f1'] = char_macro_f1
        _, _, char_weighted_f1, _ = precision_recall_fscore_support(\
                                        true_tag, pred_tag, average='weighted')
        self.summary['char_weighted_f1'] = char_weighted_f1
        
        _, _, phrase_macro_f1, _ = precision_recall_fscore_support(\
                                    phrase_binarizer.transform(true_phrases),
                                    phrase_binarizer.transform(pred_phrases),
                                    average='macro')
        self.summary['phrase_macro_f1'] = phrase_macro_f1
        _, _, phrase_weighted_f1, _ = precision_recall_fscore_support(\
                                    phrase_binarizer.transform(true_phrases),
                                    phrase_binarizer.transform(pred_phrases),
                                    average='weighted')
        self.summary['phrase_weighted_f1'] = phrase_weighted_f1
        self.summary['date'] = str(arrow.get().datetime)


    def serialize_summary(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.summary, fp, indent=2)

    def get_summary(self):
        return self.summary


    def serialize_result(self, filename): #TODO: This should use query
        with open(filename, 'w') as fp:
            json.dump(self.result_dict, fp, indent=2)
            """
            pred_list = list()
                ...: true_list = list()
                ...: sentence_list = list()
                ...: for srcid, mapping in crf_mapping.items():
                        ...:     pred_list.append(srcid)
                                     ...:     true_list.append(srcid)
                                     ...:     sentence_list.append(srcid)
                                     ...:     for c, p, t in zip(mapping['sentence'], mapping['pred_token_labels'], mapping['orig_token_labels']):
                                             ...:         pred_list.append(p)
                                                              ...:         true_list.append(t)
                                                              ...:         sentence_list.append(c)
                                                          """

    def store_result_db(self):
        #summary_query = copy(summary_query_template)
        #summary_query['source_building'] = spec['source_building']
        #summary_query['target_building'] = spec['target_building']
        #summary_query['source_sample_num'] = spec['source_sample_num']
        #summary_query['label_type'] = spec['label_type']
        #summary_query['token_type'] = spec['token_type']
        #summary_query['use_cluster_flag'] = spec['use_cluster_flag']
        #doc = copy(summary_query)
        doc = copy(self.spec)
        doc.update(self.summary)
        doc['result'] = self.result_dict
        self.result_coll.insert(doc)

    def _bilou_phraser(self, token_labels):
        phrase_labels = list()
        curr_phrase = ''
        for i, label in enumerate(token_labels):
            tag = label[0]
            if tag == 'B':
                if curr_phrase:
                # Below is redundant if the other tags handles correctly.
                    phrase_labels.append(curr_phrase)
                curr_phrase = label[2:]
            elif tag == 'I':
                if curr_phrase != label[2:]:
                    phrase_labels.append(curr_phrase)
                    curr_phrase = label[2:]
            elif tag=='L':
                if curr_phrase != label[2:]:
                    # Add if the previous label is different
                    phrase_labels.append(curr_phrase)
                # Add current label
                phrase_labels.append(label[2:])
                curr_phrase = ''
            elif tag=='O':
                # Do nothing other than pushing the previous label
                if curr_phrase:
                    phrase_labels.append(curr_phrase)
                curr_phrase = ''
            elif tag=='U':
                if curr_phrase:
                    phrase_labels.append(curr_phrase)
                phrase_labels.append(label[2:])
            else:
                print('Tag is incorrect in: {0}.'.format(label))
                try:
                    assert(False)
                except:
                    pdb.set_trace()
        return phrase_labels

