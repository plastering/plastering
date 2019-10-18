from copy import deepcopy
from multiprocessing import Pool, Manager, Process

import numpy as np
from scipy.sparse import vstack, csr_matrix, hstack, issparse, coo_matrix,\
                         lil_matrix

from .common import *

class SingleProjectClassifier():

    def __init__(self, base_classifier, mask):
        self.base_classifier = base_classifier
        self.base_mask = mask
        """
        for i, tagset in enumerate(self.binarizer.classes_):
            tags = tagset.split('_')
            mask = np.zeros(len(vectorizer.vocabulary))
#            mask = [tag for tag in vectorizer.vocabulary.values()
            for vocab, j in vectorizer.vocabulary.items():
                mask[j] = 1 if vocab in tags else 0
            self.mask_dict[i] = mask
        """


    def fit(self, X, y):
        extended_feature_dims = [i for i in range(0,X.shape[1]) \
                                 if i >= len(self.base_mask)]
        #self.mask = np.concatenate([self.base_mask, extended_feature_dims])
        mask = [i for i, v in enumerate(self.base_mask) if v == 1] +\
                range(self.base_mask, X.shape[1])
        X = X[:, mask]
        #X = X[:, np.where(self.mask==1)[0]]
        self.base_classifier.fit(X, y)

    def predict(self, X):
        assert len(self.mask) == X.shape[1]
        X = X[:, np.where(self.mask==1)[0]]
        return self.base_classifier.predict(X)

class StructuredClassifierChain():

    def __init__(self, base_classifier, binarizer, subclass_dict,
                 vocabulary_dict, n_jobs=1, use_brick_flag=False, vectorizer=None):
        self.vectorizer = vectorizer
        self.prob_flag = False
        self.use_brick_flag = use_brick_flag
        self.n_jobs = n_jobs
        self.vocabulary_dict = vocabulary_dict
        self.subclass_dict = subclass_dict
        self.base_classifier = base_classifier
        self.binarizer = binarizer
        self.upper_y_index_list = list()
        self.lower_y_index_list = list()
        self.base_classifiers = list()
        for i, tagset in enumerate(self.binarizer.classes_):
            found_upper_tagsets = find_keys(tagset, self.subclass_dict, check_in)
            upper_tagsets = [ts for ts in self.binarizer.classes_ \
                             if ts in found_upper_tagsets]
            try:
                assert len(found_upper_tagsets) == len(upper_tagsets)
            except:
                #pdb.set_trace()
                pass
            self.upper_y_index_list.append([
                np.where(self.binarizer.classes_ == ts)[0][0]
                                            for ts in upper_tagsets])
            lower_y_indices = list()
            subclasses = self.subclass_dict.get(tagset)
            if not subclasses:
                subclasses = []
            for ts in subclasses:
                indices = np.where(self.binarizer.classes_ == ts)[0]
                if len(indices)>1:
                    assert False
                elif len(indices==1):
                    lower_y_indices.append(indices[0])
            self.lower_y_index_list.append(lower_y_indices)
            self.base_classifiers.append(deepcopy(self.base_classifier))
        #self.make_proj_vec()
        self.vectorizer = None

    def make_proj_vec(self):
        vec_list = list()
        for tagset in self.binarizer.classes_:
            tags = tagset.replace('_', ' ')
            vectorized_tags = np.array([1 if v > 0 else 0 for v in
                               self.vectorizer.transform([tags]).toarray()[0]])
            vec_list.append(vectorized_tags)
        self.proj_vectors = np.vstack(vec_list)

    def _augment_X(self, X, Y):
        return np.hstack([X, Y*2])

    def _find_brick_indices(self, X, Y, orig_sample_num):
        brick_indices = list()
        for i, y in enumerate(Y):
            if i >= orig_sample_num and np.sum(y) == 1: #TODO: Need to fix orig_sample_num to consider negative samples
                brick_indices.append(i)
        return np.array(brick_indices)
        #return np.where(np.array(list(map(np.sum, Y))) == 1 )[0]


    def fit(self, X, Y, orig_sample_num = 0):
        if self.use_brick_flag:
            self.brick_indices = self._find_brick_indices(X, Y, \
                                                          orig_sample_num)
        if self.n_jobs == 1:
            return self.serial_fit(X, Y)
        else:
            return self.parallel_fit(X, Y)

    def serial_fit(self, X, Y):
        logging.info('Start fitting')
        X = self.conv_array(X)
        Y = self._augment_labels_superclasses(Y)
        for i, y in enumerate(Y.T):
            """
            sub_Y = Y[:, self.upper_y_index_list[i]]
            augmented_X = self._augment_X(X, sub_Y)
            unbiased_X, unbiased_y = self.augment_biased_sample(augmented_X, y)
            base_classifier = deepcopy(base_classifier)
            tagset = self.binarizer.classes_[i]
            tags = tagset.split('_')
            base_classifier.steps[0] = ('feature_selection', SelectKBest(chi2, k=len(tags)+3))
            try:
                self.base_classifiers[i].fit(unbiased_X, unbiased_y)
            #self.base_classifiers[i].fit(augmented_X, y)
            except:
                pass
            """
            self.base_classifiers[i] = self.sub_fit(X, Y, i)

        logging.info('Finished fitting')

    def parallel_fit(self, X,Y):
        p = Pool(self.n_jobs)
        Y = self._augment_labels_superclasses(Y)
        mapped_sub_fit = partial(self.sub_fit, X, Y)
        self.base_classifiers = p.map(mapped_sub_fit, range(0,Y.shape[1]))
        p.close()

    def augment_biased_sample(self, X, y):
        rnd_sample_num = int(X.shape[0] * 0.05)
        sub_X = X[np.where(y==1)]
        added_X_list = list()
        if sub_X.shape[0] == 0:
            return X, y
#        added_X = list()
        for i in range(0, rnd_sample_num):
            if self.use_brick_flag:
                sub_brick_indices = np.intersect1d(np.where(y==1)[0],
                                               self.brick_indices)
                if len(sub_brick_indices)==0:
                    x_1 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
                else:
                    sub_brick_X = X[sub_brick_indices]
                    x_1 = sub_brick_X[random.randint(0, sub_brick_X.shape[0]-1)]
            else:
                x_1 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
            x_2 = sub_X[random.randint(0, sub_X.shape[0] - 1)]
            avg_factor = random.random()
            #new_x = x_1 + (x_2 - x_1) * avg_factor
            if i % 2 == 0:
                new_x = x_1
            else:
                new_x = x_2
            y = np.append(y, 1)
            added_X_list.append(new_x)
        X = np.vstack([X] + added_X_list)
        assert X.shape[0] == y.shape[0]
        return X, y

    def sub_fit(self, X, Y, i):
        if i%200==0:
            logging.info('{0}th learning step'.format(i))
        if i==857:
            transformer = lambda x: [vocab for vocab, v \
                in self.vocabulary_dict.items() if x[0,v]>0]
            #pdb.set_trace()

        y = Y.T[i]
        sub_Y = Y[:, self.upper_y_index_list[i]]
        augmented_X = self._augment_X(X, sub_Y)
        unbiased_X, unbiased_y = self.augment_biased_sample(augmented_X, y)
        base_classifier = deepcopy(self.base_classifier)
        #tagset = self.binarizer.classes_[i]
        #tags = tagset.split('_')
        #base_classifier.steps[0] = ('feature_selection', SelectKBest(chi2, k=len(tags) + 3))
        try:
            base_classifier.fit(unbiased_X, unbiased_y)
            #base_classifier.fit(augmented_X, y)
        except:
            pass
        return base_classifier

    def sub_fit_proj(self, X, Y, i):
        base_base_classifier = deepcopy(self.base_classifier)
        y = Y.T[i]
        tags = self.binarizer.classes_[i].split('_')
        mask = self.proj_vectors
        base_classifier = SingleProjectClassifier(base_base_classifier, mask)
        sub_Y = Y[:, self.upper_y_index_list[i]]
        augmented_X = self._augment_X(X, sub_Y)
        base_classifier.fit(augmented_X, y)
        return base_classifier

    def parallel_fit_proj(self, X, Y):
        p = Pool(self.n_jobs)
        Y = self._augment_labels_superclasses(Y)
        mapped_sub_fit = partial(self.sub_fit_proj, X, Y)
        self.base_classifiers = p.map(mapped_sub_fit, range(0,Y.shape[1]))
        p.close()

    def serial_fit_proj(self, X, Y):
        self.base_classifiers = list()
        Y = self._augment_labels_superclasses(Y)
        for i in range(0, Y.shape[1]):
            if i%50 == 0:
                logging.info('{0}th learning step'.format(i))
            self.base_classifiers.append(self.sub_fit_proj(X, Y, i))

    def predict(self, X):
        logging.info('Start predicting')
        X = self.conv_array(X)
        Y = np.zeros((X.shape[0], len(self.binarizer.classes_)))
        for i, (upper_y_indices, base_classifier) \
                in enumerate(zip(self.upper_y_index_list,
                                 self.base_classifiers)):
            try:
                assert sum([i <= y_index for y_index in upper_y_indices]) == 0
            except:
                #pdb.set_trace()
                [y_index for y_index in upper_y_indices if y_index < i]
            sub_Y = Y[:, upper_y_indices]
            augmented_X = self._augment_X(X, sub_Y)
            if i==414 and X.shape[0]>800 and False:
                filt = base_classifier.steps[0][1]
                filtered = filt.inverse_transform(filt.transform([augmented_X[i]]))
                #print('FILTERED: ', [vocab for vocab, j in self.vocabulary_dict.items() if filtered[0][j]>0])
                #print('FROM:', [vocab for vocab, j in self.vocabulary_dict.items() if augmented_X[i][j] > 0 ])
                transformer = lambda x: [vocab for vocab, v \
                                         in self.vocabulary_dict.items() if x[v]>0]
                #pdb.set_trace() # Check why supply fan is not deteced
            try:
                if self.prob_flag:
                    prob_y = base_classifier.predict_proba(augmented_X)
                    pred_y = np.array([prob[1] for prob in prob_y])
                else:
                    pred_y = base_classifier.predict(augmented_X)
            except:
                pred_y = np.zeros(augmented_X.shape[0])
            Y[:, i] = pred_y

        if not self.prob_flag:
            Y = self._distill_Y(Y)
            
        logging.info('Finished predicting')
        return Y

    def predict_proba(self, X):
        self.prob_flag = True
        prob = self.predict(X)
        self.prob_flag = False
        return prob

    def _distill_Y(self, Y):
        logging.info('Start distilling')
        # change discharge to supply at the labels 
        # (not in the prediction but in the results)
        discharge_supply_map = dict()
        for i, tagset in enumerate(self.binarizer.classes_):
            if 'discharge' in tagset:
                discharge_supply_map[i] = np.where(self.binarizer.classes_ == \
                    tagset.replace('discharge', 'supply'))[0]
        for i_discharge, i_supply in discharge_supply_map.items():
            discharge_indices = np.where(Y[:, i_discharge] == 1)
            Y[discharge_indices, i_discharge] = 0
            try:
                Y[discharge_indices, i_supply] = 1
            except:
                pdb.set_trace()

        if self.prob_flag:
            new_Y = np.zeros(Y.shape)
            for i, y in enumerate(Y):
                new_Y[i] = np.array([1 if prob>0.5 else 0 for prob in y])
            Y = new_Y

        new_Y = deepcopy(Y)
        for i, y in enumerate(Y):
            new_y = np.zeros(len(y))
            for j, one_y in enumerate(y):
                subclass_y_indices = self.lower_y_index_list[j]
                if 1 in y[subclass_y_indices]:
                    new_y[j] = 0
                else:
                    new_y[j] = one_y
            new_Y[i] = new_y
        logging.info('Finished distilling')
        return new_Y

    def conv_array(self, d):
        if isinstance(d, np.ndarray):
            return d
        if isinstance(d, np.matrix):
            return np.asarray(d)
        else:
            return d.toarray()

    def conv_matrix(self, d):
        if isinstance(d, np.matrix):
            return d
        elif isinstance(d, np.ndarray):
            return np.matrix(d)
        else:
            return d.todense()

    def _augment_labels_superclasses(self, Y):
        logging.info('Start augmenting label mat with superclasses')
        Y = lil_matrix(Y)
        for i, vect in enumerate(Y):
            tagsets = self.binarizer.inverse_transform(vect)[0]
            updated_tagsets = reduce(adder, [
                                find_keys(tagset, self.subclass_dict, check_in)
                                for tagset in tagsets], [])
            #TODO: This is bad code. need to be fixed later.
            finished = False
            while not finished:
                try:
                    new_row = self.binarizer.transform([list(set(list(tagsets)
                                                        + updated_tagsets))])
                    finished = True
                except KeyError:
                    missing_tagset = sys.exc_info()[1].args[0]
                    updated_tagsets.remove(missing_tagset)

            Y[i] = new_row
        logging.info('Finished augmenting label mat with superclasses')
        return Y.toarray()
