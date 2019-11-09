import pdb
from collections import defaultdict

def print_sentence(srcid):
    self.sentence_dict[srcid]
from collections import OrderedDict
import json
import random
from functools import reduce

from ...metadata_interface import get_one_doc, LabeledMetadata

def adder(x, y):
    return x + y

def splitter(s):
    return s.split()




class BaseScrabble(object):
    """docstring for BaseScrabble"""
    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 building_tagsets_dict,
                 source_buildings=[],
                 source_sample_num_list=[],
                 learning_srcids=[],
                 pgid=None,
                 config={}):
        self.pgid = pgid
        self.source_buildings = source_buildings
        self.target_building = target_building
        if self.target_building not in self.source_buildings:
            self.source_buildings.append(self.target_building)
        assert len(self.source_buildings) == len(source_sample_num_list)
        self.source_sample_num_list = source_sample_num_list
        self.building_tagsets_dict = building_tagsets_dict
        self.use_brick_flag = False  # Temporarily disable it
        self.building_sentence_dict = building_sentence_dict
        self.building_label_dict = building_label_dict
        self.target_srcids = target_srcids
        self.learning_srcids = learning_srcids
        self.config = config
        self.history = []

    def leave_one_word(self, s, w):
        if w in s:
            s = s.replace(w, '')
            s = w + '-' + s
        return s

    def find_keys(self, tv, d, crit=lambda x,y:x==y):
        keys = list()
        for k, v in d.items():
            if crit(tv, v):
                keys.append(k)
        return keys

    def check_in(self, x, y):
        return x in y

    def select_random_samples_dep(
            self,
            building,
            srcids,
            n,
            use_cluster_flag,
            token_type='justseparate',
            reverse=True,
            cluster_dict=None,
            shuffle_flag=True,
            ):
        if not cluster_dict:
            cluster_filename = 'model/%s_word_clustering_%s.json' % (building.id, token_type)
            with open(cluster_filename, 'r') as fp:
                cluster_dict = json.load(fp)

        # Learning Sample Selection
        sample_srcids = set()
        length_counter = lambda x: len(x[1])
        if use_cluster_flag:
            sample_cnt = 0
            sorted_cluster_dict = OrderedDict(
                sorted(cluster_dict.items(), key=length_counter, reverse=reverse))
            #n = len(sorted_cluster_dict) #TODO: Remove if not working well
            while len(sample_srcids) < n:
                cluster_dict_items = list(sorted_cluster_dict.items())
                if shuffle_flag:
                    random.shuffle(cluster_dict_items)
                for cluster_num, srcid_list in cluster_dict_items:
                    valid_srcid_list = set(srcid_list)\
                            .intersection(set(srcids))\
                            .difference(set(sample_srcids))
                    if len(valid_srcid_list) > 0:
                        sample_srcids.add(\
                                random.choice(list(valid_srcid_list)))
                    if len(sample_srcids) >= n:
                        break
        else:
            sample_srcids = random.sample(srcids, n)
        return list(sample_srcids)

    def alpha_tokenizer(s): 
        return re.findall('[a-zA-Z]+', s)

    def hier_clustering(d, threshold=3):
        srcids = d.keys()
        tokenizer = lambda x: x.split()
        vectorizer = TfidfVectorizer(tokenizer=tokenizer)
        assert isinstance(d, dict)
        assert isinstance(list(d.values())[0], list)
        assert isinstance(list(d.values())[0][0], str)
        doc = [' '.join(d[srcid]) for srcid in srcids]
        vect = vectorizer.fit_transform(doc)
        #TODO: Make vect aligned to the required format
        z = linkage(vect.toarray(), metric='cityblock', method='complete')
        dists = list(set(z[:,2]))
#    threshold = 3
        #threshold = (dists[2] + dists[3]) / 2
        b = hier.fcluster(z, threshold, criterion='distance')
        cluster_dict = defaultdict(list)
        for srcid, cluster_id in zip(srcids, b):
            cluster_dict[str(cluster_id)].append(srcid)
        value_lengther = lambda x: len(x[1])
        return OrderedDict(\
                   sorted(cluster_dict.items(), key=value_lengther, reverse=True))

    def set_logger(logfile=None):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(message)s')

        # Console Handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

        # File Handler
        if logfile:
            fh = logging.FileHandler(logfile, mode='w+')
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
        return logger


    def parallel_func(orig_func, return_idx, return_dict, *args):
        return_dict[return_idx] = orig_func(*args)


    def iteration_wrapper(self, iter_num, func, prev_data=None, *params):
        step_datas = list()
        if not prev_data:
            prev_data = {'iter_num':0,
                         'learning_srcids': [],
                         'model_uuid': None}
        for i in range(0, iter_num):
            print('{0} th stage started'.format(prev_data['iter_num']))
            step_data = func(prev_data, *params)
            print('{0} th stage finished'.format(prev_data['iter_num']))
            step_datas.append(step_data)
            prev_data = step_data
            prev_data['iter_num'] += 1
        return step_datas


    def replace_num_or_special(word):
        if re.match('\d+', word):
            return 'NUMBER'
        elif re.match('[a-zA-Z]+', word):
            return word
        else:
            return 'SPECIAL'


    def get_cluster_dict(building):
        cluster_filename = 'model/%s_word_clustering_justseparate.json' % (building.id)
        with open(cluster_filename, 'r') as fp:
            cluster_dict = json.load(fp)
        return cluster_dict

    def get_label_dict(building):
        filename = 'metadata/%s_label_dict_justseparate.json' % (building.id)
        with open(filename, 'r') as fp:
            data = json.load(fp)
        return data

    def print_sentence(self, srcid):
        print(''.join(self.sentence_dict[srcid]))

    def print_pred(self, preds, srcids):
        for srcid in srcids:
            sentence = self.sentence_dict[srcid]
            char_labels = self.label_dict[srcid]
            for char_label, char_pred, char \
                    in zip(char_labels, preds[srcid], sentence):
                is_correct = char_label == char_pred
                print('{0}: {1}\t{2}\t{3}'.format('O' if is_correct else 'X',
                                                  char_pred, char_label, char))

    def get_learning_sample_nums(self):
        num_samples = defaultdict(int)
        for srcid in self.learning_srcids:
            for building, sentences in self.building_sentence_dict.items():
                if srcid in sentences:
                    num_samples[building.id] += 1
                    break
        assert sum(num_samples.values()) == len(set(self.learning_srcids))
        return num_samples
