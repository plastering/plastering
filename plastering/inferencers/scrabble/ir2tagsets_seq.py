from uuid import uuid4
from operator import itemgetter
from itertools import chain
from copy import deepcopy
import json
from functools import reduce
def gen_uuid():
    return str(uuid4())

import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support

from skmultilearn.problem_transform import LabelPowerset, ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance

from scipy.sparse import vstack, csr_matrix, hstack, issparse, coo_matrix
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.layers import Input
from keras.optimizers import RMSprop
from keras import regularizers
from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector, Dropout
from keras.layers.core import Dense, Activation
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.layers import Input
from keras.models import Sequential

from time_series_to_ir import TimeSeriesToIR
from base_scrabble import BaseScrabble
from common import *
from hcc import StructuredClassifierChain
from brick_parser import pointTagsetList        as  point_tagsets,\
                         locationTagsetList     as  location_tagsets,\
                         equipTagsetList        as  equip_tagsets,\
                         pointSubclassDict      as  point_subclass_dict,\
                         equipSubclassDict      as  equip_subclass_dict,\
                         locationSubclassDict   as  location_subclass_dict,\
                         tagsetTree             as  tagset_tree

tagset_list = point_tagsets + location_tagsets + equip_tagsets
tagset_list.append('networkadapter')


def tree_flatter(tree, init_flag=True):
    branches_list = list(tree.values())
    d_list = list(tree.keys())
    for branches in branches_list:
        for branch in branches:
            added_d_list = tree_flatter(branch)
            d_list = [d for d in d_list if d not in added_d_list]\
                    + added_d_list
    return d_list

def extend_tree(tree, k, d):
    for curr_head, branches in tree.items():
        if k==curr_head:
            branches.append(d)
        for branch in branches:
            extend_tree(branch, k, d)

def calc_leaves_depth(tree, d=dict(), depth=0):
    curr_depth = depth + 1
    for tagset, branches in tree.items():
        if d.get(tagset):
            d[tagset] = max(d[tagset], curr_depth)
        else:
            d[tagset] = curr_depth
        for branch in branches:
            new_d = calc_leaves_depth(branch, d, curr_depth)
            for k, v in new_d.items():
                if d.get(k):
                    d[k] = max(d[k], v)
                else:
                    d[k] = v
    return d

def augment_tagset_tree(tagsets, subclass_dict):
    for tagset in set(tagsets):
        if '-' in tagset:
            classname = tagset.split('-')[0]
            #tagset_tree[classname].append({tagset:[]})
            extend_tree(tagset_tree, classname, {tagset:[]})
            subclass_dict[classname].append(tagset)
            subclass_dict[tagset] = []
        else:
            if tagset not in subclass_dict.keys():
                classname = tagset.split('_')[-1]
                subclass_dict[classname].append(tagset)
                subclass_dict[tagset] = []
                extend_tree(tagset_tree, classname, {tagset:[]})



class SequenceAutoencoder(object):
    def __init__(self, latent_dim=20):
        self.latent_dim = latent_dim

    def fit_new(self, x, y=None):
        timesteps = x.shape[1]
        input_dim = x.shape[2]
        self.ae = Sequential()
        self.ae.add(Dense(self.latent_dim,
                    input_shape=(timesteps,input_dim,),
                    activation='relu',
                    name='enc'))
        self.ae.add(Dropout(0.2))
        self.ae.add(Dense(input_dim,
                    activation='softmax',
                    name='dec'))

        self.encoder = Model(inputs=self.ae.input,
                             outputs=self.ae.get_layer('enc').output)
        #rmsprop = RMSprop(lr=0.05)
        self.ae.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'],)
        self.ae.fit(x, x, epochs=1)

    def fit(self, x, y=None):
        timesteps = x.shape[1]
        input_dim = x.shape[2]
        self.ae = Sequential()
        #m.add(LSTM(latent_dim, input_dim=in_dim, return_sequen|ces=True, name='enc'), )
        self.ae.add(LSTM(self.latent_dim,
                         activation='softsign',
                         input_shape=(timesteps,input_dim,),
                         return_sequences=True,
                         unroll=True,
                         name='enc'), )
        self.ae.add(LSTM(input_dim,
                         activation='softsign',
                         return_sequences=True,
                         unroll=True,
                         name='dec',
                         ))
        self.ae.add(Activation('softmax'))

        self.encoder = Model(inputs=self.ae.input,
                             outputs=self.ae.get_layer('enc').output)
        rmsprop = RMSprop(lr=0.005)
        self.ae.compile(loss='categorical_hinge',
                  optimizer=rmsprop,
                  metrics=['categorical_accuracy', 'binary_accuracy'],)
        self.ae.fit(x, x, epochs=1)

    def fit_dep(self, x, y=None):
        timesteps = x.shape[1]
        input_dim = x.shape[2]
        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(self.latent_dim)(inputs)

        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        encoded_input = Input(shape=(self.latent_dim,))

        self.sequence_autoencoder = Model(inputs, decoded)
        self.encoder = Model(inputs, encoded)

        self.sequence_autoencoder.compile(
            #loss='binary_crossentropy',
            loss='categorical_crossentropy',
            optimizer='RMSprop',
            metrics=['binary_accuracy']
        )
        self.sequence_autoencoder.fit(x, x)

    def transform(self, x):
        return self.encoder.predict(x)

class SequenceVectorizer(object):
    def __init__(self, tokenizer, vocabulary, max_len):
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.max_len = max_len

    def fit(self, x):
        #le = LabelBinarizer().fit(reduce(adder , x))
        self.le = LabelBinarizer().fit(list(self.vocabulary.keys()))
        vocabs = list(self.vocabulary.keys())
        binarized = self.le.transform(vocabs)
        (locs, indices) = np.where(binarized==1)
        for loc, index in zip(locs, indices):
            self.vocabulary[vocabs[loc]] = index

    def transform(self, x):
        x = map(self.tokenizer, x)
        stack = []
        for sentence in x:
            encoded = self.le.transform(sentence)
            padder = np.zeros((self.max_len - encoded.shape[0],
                               encoded.shape[1]))
            encoded = np.vstack([encoded, padder])
            stack.append(encoded)
        encoded_labels = np.stack(stack)
        return encoded_labels

class Ir2Tagsets(BaseScrabble):

    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 building_tagsets_dict,
                 source_buildings=[],
                 source_sample_num_list=[],
                 learning_srcids=[],
                 conf={}):
        super(Ir2Tagsets, self).__init__(
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 building_tagsets_dict,
                 source_buildings,
                 source_sample_num_list,
                 learning_srcids,
                 conf)
        self.ts2ir = None
        self.ts_feature_filename = 'temp/features.pkl'

        if 'use_cluster_flag' in conf:
            self.use_cluster_flag = conf['use_cluster_flag']
        else:
            self.use_cluster_flag = True
        if 'eda_flag' in conf:
            self.eda_flag = conf['eda_flag'],
        else:
            self.eda_flag = False
        if 'use_brick_flag' in conf:
            self.use_brick_flag = conf['use_brick_flag']
        else:
            self.use_brick_flag = True
        if 'n_jobs' in conf:
            self.n_jobs = conf['n_jobs']
        else:
            #self.n_jobs = 1
            self.n_jobs = 24
        if 'ts_flag' in conf:
            self.ts_flag = conf['ts_flag']
        else:
            self.ts_flag = False
        if 'negative_flag' in conf:
            self.negative_flag = conf['negative_flag']
        else:
            self.negative_flag = False
        if 'tagset_classifier_type' in conf:
            self.tagset_classifier_type = conf['tagset_classifier_type']
        else:
            #self.tagset_classifier_type = 'StructuredCC_autoencoder'
            self.tagset_classifier_type = 'StructuredCC'
        if 'n_estimators' in conf:
            self.n_estimators = conf['n_estimators']
        else:
            self.n_estimators = 10 # TODO: Find the proper value
        if 'vectorizer_type' in conf:
            self.vectorizer_type = conf['vectorizer_type']
        else:
            self.vectorizer_type = 'tfidf'
            #self.vectorizer_type = 'sequence'
        if 'query_strategy' in conf:
            self.query_strategy = conf['query_strategy']
        else:
            self.query_strategy = 'phrase_util'
        if 'autoencode' in conf:
            self.autoencode = conf['autoencode']
        else:
            #self.autoencode = True
            self.autoencode = False
        self._init_data()
        self._init_brick()

    def _init_brick(self):
        self.brick_srcids = []
        self.tagset_list = point_tagsets + location_tagsets + equip_tagsets
        self.tagset_list.append('networkadapter')

        self.subclass_dict = dict()
        self.subclass_dict.update(point_subclass_dict)
        self.subclass_dict.update(equip_subclass_dict)
        self.subclass_dict.update(location_subclass_dict)
        self.subclass_dict['networkadapter'] = list()
        self.subclass_dict['unknown'] = list()
        self.subclass_dict['none'] = list()
        self.tagset_tree = deepcopy(tagset_tree)


    def _init_data(self):
        self.sentence_dict = {}
        self.label_dict = {}
        self.tagsets_dict = {}
        self.phrase_dict = {}
        self.point_dict = {}
        self.max_len = None

        for building, source_sample_num in zip(self.source_buildings,
                                               self.source_sample_num_list):
            self.sentence_dict.update(self.building_sentence_dict[building])
            one_label_dict = self.building_label_dict[building]
            self.label_dict.update(one_label_dict)

            if not self.learning_srcids:
                sample_srcid_list = select_random_samples(
                                        building,
                                        one_label_dict.keys(),
                                        source_sample_num,
                                        self.use_cluster_flag)
                self.learning_srcids += sample_srcid_list
            one_tagsets_dict = self.building_tagsets_dict[building]
            self.tagsets_dict.update(one_tagsets_dict)
            for srcid, tagsets in one_tagsets_dict.items():
                point_tagset = 'none'
                for tagset in tagsets:
                    if tagset in point_tagsets:
                        point_tagset = tagset
                        break
                self.point_dict[srcid] = point_tagset

        self.phrase_dict = make_phrase_dict(self.sentence_dict,
                                            self.label_dict)
        self.max_len = max([len(phrases) for phrases
                            in self.phrase_dict.values()])
        # validation
        for srcid in self.target_srcids:
            assert srcid in self.tagsets_dict

    def _augment_brick_samples(self, doc, srcids):
        brick_truths_dict = dict()
        self.brick_srcids = []
        brick_doc = []
        logging.info('Start adding Brick samples')
        brick_copy_num = 6
        self.brick_tagsets_dict = dict()
        self.brick_doc = list()
        for tagset in tagset_list:
            for j in range(0, brick_copy_num):
                #multiplier = random.randint(2, 6)
                srcid = 'brick;' + gen_uuid()
                self.brick_srcids.append(srcid)
                self.brick_tagsets_dict[srcid] = [tagset]
                tagset_doc = list()
                for tag in tagset.split('_'):
                    tagset_doc += [tag] * random.randint(1,2)
                brick_doc.append(' '.join(tagset_doc))
        doc += brick_doc
        self.tagsets_dict.update(self.brick_tagsets_dict)
        srcids += self.brick_srcids
        return doc, srcids


    def _extend_tagset_list(self, new_tagsets):
        self.tagset_list += new_tagsets
        self.tagset_list = list(set(self.tagset_list))

    def update_model(self, srcids):
        self.learning_srcids += srcids
        self.target_srcids = [srcid for srcid in self.target_srcids
                              if srcid not in self.learning_srcids]
        invalid_num = sum([srcid not in self.tagsets_dict for srcid in
                           self.learning_srcids + self.target_srcids]) #debug
        self._extend_tagset_list(reduce(adder, [self.tagsets_dict[srcid]
            for srcid in self.learning_srcids + self.target_srcids]))
        augment_tagset_tree(self.tagset_list, self.subclass_dict)
        self._build_tagset_classifier(self.learning_srcids,
                                      self.target_srcids,
                                      validation_srcids=[])

    def _make_doc_vectorizer(self, doc):
        doc = [sentence.split() for sentence in doc]
        le = LabelBinarizer().fit(reduce(adder , doc))
        stack = []
        for sentence in doc:
            encoded = le.transform(sentence)
            padder = np.zeros((self.max_len - encoded.shape[0],
                               encoded.shape[1]))
            encoded = np.vstack([encoded, padder])
            stack.append(encoded)
        encoded_labels = np.stack(stack)
        return encoded_labels

    def _augment_negative_examples(self, doc, srcids):
        negative_doc = []
        negative_truths_dict = {}
        negative_srcids = []
        for srcid in self.learning_srcids:
            true_tagsets = list(set(self.tagsets_dict[srcid]))
            sentence = self.phrase_dict[srcid]
            for tagset in true_tagsets:
                negative_srcid = srcid + ';' + gen_uuid()
                removing_tagsets = set()
                new_removing_tagsets = set([tagset])
                removing_tags = []
                negative_tagsets = list(filter(tagset.__ne__, true_tagsets))
                i = 0
                while len(new_removing_tagsets) != len(removing_tagsets):
                    i += 1
                    if i>5:
                        pdb.set_trace()
                    removing_tagsets = deepcopy(new_removing_tagsets)
                    for removing_tagset in removing_tagsets:
                        removing_tags += removing_tagset.split('_')
                    for negative_tagset in negative_tagsets:
                        for tag in removing_tags:
                            if tag in negative_tagset.split('_'):
                                new_removing_tagsets.add(negative_tagset)
                negative_sentence = [tag for tag in sentence if\
                                     tag not in removing_tags]
                for tagset in removing_tagsets:
                    negative_tagsets = list(filter(tagset.__ne__,
                                                   negative_tagsets))

    #            negative_sentence = [word for word in sentence \
    #                                 if word not in tagset.split('_')]
                negative_doc.append(' '.join(negative_sentence))
                negative_truths_dict[negative_srcid] = negative_tagsets
                negative_srcids.append(negative_srcid)
        for i in range(0,50):
            # Add empty examples
            negative_srcid = gen_uuid()
            negative_doc.append('')
            negative_srcids.append(negative_srcid)
            negative_truths_dict[negative_srcid] = []
        doc += negative_doc
        srcids += negative_srcids
        self.tagsets_dict.update(negative_truths_dict)
        return doc, srcids

    def _build_tagset_classifier(self,
                                 learning_srcids,
                                 target_srcids,
                                 validation_srcids):

        learning_srcids = deepcopy(learning_srcids)

        # Update TagSet pool to include TagSets not in Brick.
        orig_sample_num = len(learning_srcids)
        new_tagset_list = tree_flatter(self.tagset_tree, [])
        new_tagset_list = new_tagset_list + [ts for ts in self.tagset_list \
                                             if ts not in new_tagset_list]
        self.tagset_list = new_tagset_list
        self.tagset_binarizer = MultiLabelBinarizer(self.tagset_list)
        self.tagset_binarizer.fit([self.tagset_list])
        assert self.tagset_list == self.tagset_binarizer.classes_.tolist()

        #self.tagsets_dict = {srcid: self.tagsets_dict[srcid]
        #                         for srcid in learning_srcids}


        ## Init brick tag_list
        # TODO: Maybe this should be done in initialization stage.
        self.tag_list = list(set(reduce(adder, map(splitter,
                                                   self.tagset_list))))

        # All possible vocabularies.
        vocab_dict = dict([(tag, i) for i, tag in enumerate(self.tag_list)])

        # Define Vectorizer
        tokenizer = lambda x: x.split()
        # TODO: We could use word embedding like word2vec here instead.
        if self.vectorizer_type == 'tfidf':
            self.tagset_vectorizer = TfidfVectorizer(tokenizer=tokenizer,
                                                vocabulary=vocab_dict)
        elif self.vectorizer_type == 'meanbembedding':
            self.tagset_vectorizer = MeanEmbeddingVectorizer(tokenizer=tokenizer,
                                                        vocabulary=vocab_dict)
        elif self.vectorizer_type == 'count':
            self.tagset_vectorizer = CountVectorizer(tokenizer=tokenizer,
                                                vocabulary=vocab_dict)
        elif self.vectorizer_type == 'sequence':
            self.tagset_vectorizer = SequenceVectorizer(tokenizer=tokenizer,
                                                        vocabulary=vocab_dict,
                                                        max_len=self.max_len)
        else:
            raise Exception('Wrong vectorizer type: {0}'
                                .format(self.vectorizer_type))

        if self.ts_flag:
            pass
            #TODO: Run self._augment_with_ts()

        ## Transform learning samples
        learning_doc = [' '.join(self.phrase_dict[srcid])
                        for srcid in learning_srcids]
        test_doc = [' '.join(self.phrase_dict[srcid])
                    for srcid in target_srcids]

        ## Augment with negative examples.
        if self.negative_flag:
            learning_doc, learning_srcids  = \
                self._augment_negative_examples(learning_doc, learning_srcids)


        ## Init Brick samples.
        if self.use_brick_flag:
            learning_doc, learning_srcids  = \
                self._augment_brick_samples(learning_doc,
                                            learning_srcids)

        self.tagset_vectorizer.fit(learning_doc + test_doc)# + brick_doc)

        # Apply Easy-Domain-Adaptation mechanism. Not useful.
        if self.eda_flag:
            pass
            # TODO: self._augment_eda()
        else:
            # Make TagSet vectors.
            learning_vect_doc = self.tagset_vectorizer.transform(learning_doc)
            if not isinstance(learning_vect_doc, np.ndarray):
                learning_vect_doc = learning_vect_doc.todense()

        truth_mat = csr_matrix([self.tagset_binarizer.transform(
                                    [self.tagsets_dict[srcid]])[0]
                                for srcid in learning_srcids])
        if self.eda_flag:
            assert False, 'This should not be called for now'
            zero_vectors = self.tagset_binarizer.transform(\
                        [[] for i in range(0, unlabeled_vect_doc.shape[0])])
            truth_mat = vstack([truth_mat, zero_vectors])
            learning_vect_doc = np.vstack([learning_vect_doc, unlabeled_vect_doc])

        if self.autoencode:
            self.encoder = SequenceAutoencoder()
            self.encoder .fit(learning_vect_doc)
            learning_vect_doc = self.encoder.transform(learning_vect_doc)

        if self.tagset_classifier_type == 'StructuredCC_autoencoder':
            def meta_scc(**kwargs):
                #feature_selector = SelectFromModel(LinearSVC(C=1))
                #feature_selector = SequenceAutoencoder()
                base_base_classifier = GradientBoostingClassifier(**kwargs)
                base_classifier = Pipeline([#('feature_selection',
                                            # feature_selector),
                                            ('classification',
                                             base_base_classifier)
                                           ])
                tagset_classifier = StructuredClassifierChain(
                                    base_classifier,
                                    self.tagset_binarizer,
                                    self.subclass_dict,
                                    self.tagset_vectorizer.vocabulary,
                                    self.n_jobs,
                                    self.use_brick_flag,
                                    self.tagset_vectorizer)
                return tagset_classifier

            meta_classifier = meta_scc
            """
            rf_params_list_dict = {
                'n_estimators': [10, 50, 100],
                'criterion': ['gini', 'entropy'],
                'max_features': [None, 'auto'],
                'max_depth': [1, 5, 10, 50],
                'min_samples_leaf': [2,4,8],
                'min_samples_split': [2,4,8]
            }
            gb_params_list_dict = {
                'loss': ['deviance', 'exponential'],
                'learning_rate': [0.1, 0.01, 1, 2],
                'criterion': ['friedman_mse', 'mse'],
                'max_features': [None, 'sqrt'],
                'max_depth': [1, 3, 5, 10],
                'min_samples_leaf': [1,2,4,8],
                'min_samples_split': [2,4,8]
            }
            params_list_dict = gb_params_list_dict
            """
        elif self.tagset_classifier_type == 'StructuredCC':
            def meta_scc(**kwargs):
                feature_selector = SelectFromModel(LinearSVC(C=1))
                #feature_selector = SelectFromModel(LinearSVC(C=0.01, penalty='l1', dual=False))
                base_base_classifier = GradientBoostingClassifier(**kwargs)
                #base_base_classifier = SGDClassifier(loss='modified_huber', penalty='elasticnet')
                #base_base_classifier = PassiveAggressiveClassifier(loss='squared_hinge', C=0.1)
                #base_base_classifier = LogisticRegression()
                #base_base_classifier = RandomForestClassifier(**kwargs)
                base_classifier = Pipeline([('feature_selection',
                                             feature_selector),
                                            ('classification',
                                             base_base_classifier)
                                           ])
                tagset_classifier = StructuredClassifierChain(
                                    base_classifier,
                                    self.tagset_binarizer,
                                    self.subclass_dict,
                                    self.tagset_vectorizer.vocabulary,
                                    self.n_jobs,
                                    self.use_brick_flag,
                                    self.tagset_vectorizer)
                return tagset_classifier
            meta_classifier = meta_scc
            """
            rf_params_list_dict = {
                'n_estimators': [10, 50, 100],
                'criterion': ['gini', 'entropy'],
                'max_features': [None, 'auto'],
                'max_depth': [1, 5, 10, 50],
                'min_samples_leaf': [2,4,8],
                'min_samples_split': [2,4,8]
            }
            gb_params_list_dict = {
                'loss': ['deviance', 'exponential'],
                'learning_rate': [0.1, 0.01, 1, 2],
                'criterion': ['friedman_mse', 'mse'],
                'max_features': [None, 'sqrt'],
                'max_depth': [1, 3, 5, 10],
                'min_samples_leaf': [1,2,4,8],
                'min_samples_split': [2,4,8]
            }
            params_list_dict = gb_params_list_dict
            """
        else:
            assert False, 'Not existing classifier type: {0}'\
                .format(self.tagset_classifier_type)
        best_params = {'learning_rate':0.1, 'subsample':0.25}
        self.tagset_classifier  = meta_classifier(**best_params)

        # Actual fitting.
        if isinstance(self.tagset_classifier, StructuredClassifierChain):
            self.tagset_classifier.fit(learning_vect_doc, truth_mat.toarray(), \
                                  orig_sample_num=len(learning_vect_doc)
                                  - len(self.brick_srcids))
        else:
            assert False, 'This should not be reachable for now'
            self.tagset_classifier.fit(learning_vect_doc, truth_mat.toarray())

    def ir2tagset_al_query_samples_phrase_util(self,
                                               test_srcids,
                                               building,
                                               pred_tagsets_dict,
                                               inc_num):
        phrase_usage_dict = {}
        for srcid in test_srcids:
            pred_tagsets = pred_tagsets_dict[srcid]
            phrase_usage_dict[srcid] = self._determine_used_phrases(
                                           self.phrase_dict[srcid],
                                           pred_tagsets)

        phrase_usages = list(phrase_usage_dict.values())
        mean_usage_rate = np.mean(phrase_usages)
        std_usage_rate = np.std(phrase_usages)
        # Select underexploited sentences.
        threshold = mean_usage_rate - std_usage_rate
        todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(
                                   self.sentence_dict[srcid])))
                                   for srcid, usage_rate
                                   in phrase_usage_dict.items()
                                   if usage_rate < threshold and srcid in test_srcids)
        cluster_dict = get_cluster_dict(building)
        todo_srcids = select_random_samples(building, \
                              list(todo_sentence_dict.keys()),
                              min(inc_num, len(todo_sentence_dict)), \
                              True,\
                              reverse=True,
                              cluster_dict=cluster_dict,
                              shuffle_flag=False
                             )
        #if the numbers are not enough randomly select more:
        if len(todo_srcids) < inc_num:
            more_num = inc_num - len(todo_srcids)
            todo_sentence_dict = dict((srcid, alpha_tokenizer(''.join(
                                       self.sentence_dict[srcid])))
                                       for srcid, usage_rate
                                       in phrase_usage_dict.items()
                                       if srcid in test_srcids)
            cluster_dict = get_cluster_dict(building)
            todo_srcids = select_random_samples(building, \
                                  list(todo_sentence_dict.keys()),
                                  min(more_num, len(todo_sentence_dict)), \
                                  True,\
                                  cluster_dict=cluster_dict,
                                  shuffle_flag=True
                                 )
        return todo_srcids

    def select_informative_samples(self, sample_num):
        pred = self.predict(self.target_srcids)
        if self.query_strategy == 'phrase_util':
            new_srcids = self.ir2tagset_al_query_samples_phrase_util(
                                self.target_srcids,
                                self.target_building,
                                pred,
                                sample_num)
        else:
            raise ValueError('Query Strategy Wrong: {0}'.format(query_strategy))
        return new_srcids

    def _determine_used_phrases(self, phrases, tagsets):
        phrases_usages = list()
        pred_tags = reduce(adder, [tagset.split('_') for tagset in tagsets], [])
        used_cnt = 0.0
        unused_cnt = 0.0
        for phrase in phrases:
            phrase_tags = phrase.split('_')
            for tag in phrase_tags:
                if tag in ['leftidentifier', 'rightidentifier']:
                    continue
                if tag in pred_tags:
                    used_cnt += 1 / len(phrase_tags)
                else:
                    unused_cnt += 1 / len(phrase_tags)
        if used_cnt == 0:
            score = 0
        else:
            score = used_cnt / (used_cnt + unused_cnt)
        return score

    def _predict_and_proba(self, target_srcids):
        #return self.tagset_classifier, self.tagset_vectorizer, self.tagset_binarizer, self.ts2ir
        phrase_dict = {srcid: self.phrase_dict[srcid]
                       for srcid in target_srcids}
        if self.ts_flag:
            phrase_dict = self._augment_phrases_with_ts(phrase_dict, target_srcids, self.ts2ir)
        doc = [' '.join(phrase_dict[srcid]) for srcid in target_srcids]
        vect_doc = self.tagset_vectorizer.transform(doc) # should this be fit_transform?
        if self.autoencode:
            vect_doc = self.encoder.transform(vect_doc)


        certainty_dict = dict()
        tagsets_dict = dict()
        pred_mat = self.tagset_classifier.predict(vect_doc)
        prob_mat = self.tagset_classifier.predict_proba(vect_doc)
        if not isinstance(pred_mat, np.ndarray):
            try:
                pred_mat = pred_mat.toarray()
            except:
                pred_mat = np.asarray(pred_mat)
        pred_tagsets_dict = dict()
        pred_certainty_dict = dict()
        pred_point_dict = dict()
        for i, (srcid, pred) in enumerate(zip(target_srcids, pred_mat)):
        #for i, (srcid, pred, point_pred) \
                #in enumerate(zip(srcids, pred_mat, point_mat)):
            pred_tagsets_dict[srcid] = self.tagset_binarizer.inverse_transform(\
                                            np.asarray([pred]))[0]
            #pred_tagsets_dict[srcid] = list(binarizer.inverse_transform(pred)[0])
            #pred_point_dict[srcid] = point_tagsets[point_pred]
            #pred_vec = [prob[i][0] for prob in prob_mat]
            #pred_certainty_dict[srcid] = pred_vec
            pred_certainty_dict[srcid] = 0
        pred_certainty_dict = OrderedDict(sorted(pred_certainty_dict.items(), \
                                                 key=itemgetter(1), reverse=True))
        logging.info('Finished prediction')
        return pred_tagsets_dict, pred_certainty_dict

    def predict(self, target_srcids=None):
        if not target_srcids:
            target_srcids =self.target_srcids
        pred, _ =self._predict_and_proba(target_srcids)
        return pred

    def predict_proba(self, target_srcids=None):
        if not target_srcids:
            target_srcids =self.target_srcids
        _, proba =self._predict_and_proba(target_srcids)
        return proba

    def evaluate(self, pred):
        #set_sim = accuracy_func(pred_tagsets, true_tagsets, labels=None):
        set_sim = None
        macro_f1 = None
        res = {
            'set_similarity': set_sim,
            'macro_f1': macro_f1
        }
