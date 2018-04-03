import matplotlib
matplotlib.use('Agg')

import operator
import re
import json
import csv
import random
import pdb
from collections import defaultdict
from copy import deepcopy

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import arrow

import scipy
from scipy.cluster.vq import *
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as hier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

from . import plotter

listdefaultdict = lambda : defaultdict(list)
DEBUG = True 

def tokenizer(s):
    return re.findall('[a-z]+', s.lower())


class Zodiac(object):
    """docstring for Zodiac"""

    def select_informative_samples_only(self, sample_num=10):
        new_srcids = []
        new_sample_cnt = 0

        for p in self.cluster_map.keys(): # for each cluster

            # Escape if already labeled.
            if p in self.labeled_clusters:
                continue

            # Get sensors from this cluster.
            sample_bow = []
            for k in self.cluster_map[p]['sensor_ids']:
                sample_bow.append(self.bow_array[k])
            sample_bow = np.array(sample_bow)

            # Apply trained model:
            confidence = self.model.predict_proba(sample_bow)
            prediction_label = self.model.predict(sample_bow)
            # Get overall max confidence for any sensor in cluster:
            max_c = 0
            for c in confidence:
                max_c = max(np.append(c,[max_c]))

            # Compare with Thresholds.
            flag = 0
            if max_c < self.T_low:
                flag = 1
            if max_c > self.T_high:
                flag = 2

            if flag==1:
                if new_sample_cnt >= sample_num:
                    continue
                new_sample_cnt += 1
                new_srcid = random.choice([sensor['source_id'] for sensor 
                                           in self.cluster_map[p]['sensors']])
                new_srcids.append(new_srcid)
                self.labeled_clusters.append(p)

        return new_srcids

    def select_informative_samples(self, sample_num=10):

        # Iteratively apply Random Forest to label new sensors
        begin_time = arrow.get()

        self.change_thresholds = True
        n_wrong_confident_sensor_pred = 0
        self.sensor_bow = list(self.sensor_bow)
        self.sensor_labels = list(self.sensor_labels)
        n_high_confidence_sensors = 0
        n_manually_labeled_thisepoch = 0 # epoch = whatever happens after (re-) training RF models
        new_sample_cnt = 0

        for p in self.cluster_map.keys(): # for each cluster
        #for p in sorted_equip_keys:
            #p = p[0]

            # Escape if already labeled.
            if p in self.labeled_clusters:
                continue


            # Get sensors from this cluster.
            sample_bow = []
            for k in self.cluster_map[p]['sensor_ids']:
                sample_bow.append(self.bow_array[k])
            sample_bow = np.array(sample_bow)


            # Apply trained model:
            confidence = self.model.predict_proba(sample_bow)
            prediction_label = self.model.predict(sample_bow)
            # Get overall max confidence for any sensor in cluster:
            max_c = 0
            for c in confidence:
                max_c = max(np.append(c,[max_c]))


            # Compare with Thresholds.
            flag = 0
            if max_c < self.T_low:
                flag = 1
            if max_c > self.T_high:
                flag = 2

            if flag==1:
                if new_sample_cnt >= sample_num:
                    continue
                n_manually_labeled_thisepoch+=1
                new_sample_cnt += 1


            # Handle the cluster beyond threshold:
            if flag > 0:
                self.change_thresholds = False
                self.labeled_clusters.append(p)

                # For each sensor in this cluster:
                for k in range(len(self.cluster_map[p]['sensors'])):
                    sourceid = self.cluster_map[p]['sensors'][k]['source_id']
                    true_type = self.true_sensor_types[sourceid]
                    pred_type = prediction_label[k]

                    if flag==2:
                        n_high_confidence_sensors+=1
                        if pred_type != true_type:
                            n_wrong_confident_sensor_pred+=1

                        # append these sensors into labeled ones (with possibly wrong labels):
                        self.sensor_bow.append(
                            self.bow_array[self.cluster_map[p]['sensor_ids'][k]])
                        self.sensor_labels.append(pred_type)

                    if flag==1:
                        # append these sensors into labeled ones (with ground truth):
                        self.sensor_bow.append(
                            self.bow_array[self.cluster_map[p]['sensor_ids'][k]])
                        self.sensor_labels.append(true_type)
                break
        """
        if new_sample_cnt < sample_num:
            print("only {0} samples were added instead of {1}"\
                      .format(new_sample_cnt, sample_num))
        """
        return n_manually_labeled_thisepoch, \
               n_wrong_confident_sensor_pred, \
               n_high_confidence_sensors, \
               len(self.sensor_labels), \
               100.0 * len(self.sensor_labels)/len(self.bow_array),\
               self.change_thresholds,\
               len(self.bow_array) - len(self.sensor_labels), \
               new_sample_cnt

    def __init__(self, names, descs, units,
              type_strs, types, jci_names, true_sensor_types, conf={}):
        super(Zodiac, self).__init__()
        if conf:
            self.conf = conf
        else:
            conf = {
                'n_estimators': 400,
                'random_state': 0,
                #'n_jobs': 1
                }
        self.model = RandomForestClassifier(n_estimators=conf['n_estimators'],
                                            random_state=conf['random_state'],
                                            #n_jobs=conf['n_jobs']
                                            )
        # init data
        self.learned_srcids = []
        self.srcids = list(names.keys())
        self.true_sensor_types = true_sensor_types
        self.sensors = [{'source_id': srcid,
                    'name': names[srcid],
                    'description': descs[srcid],
                    'unit': units[srcid],
                    'type_string': type_strs[srcid],
                    'type': types[srcid],
                    'jci_name': jci_names[srcid]
                    } for srcid in self.srcids]
        sensor_df = pd.DataFrame(self.sensors)
        sensor_df = sensor_df.set_index('source_id')
        sensor_df = sensor_df.groupby(sensor_df.index).first()
        print(len(self.sensors))
        names_list = [names[srcid] for srcid in self.srcids]
        desc_list = [descs[srcid] for srcid in self.srcids]
        unit_list = [units[srcid] for srcid in self.srcids]
        unit_list = [0 if unit == '' else unit for unit in unit_list]
        type_str_list = [type_strs[srcid] for srcid in self.srcids]
        #type_list = [types[srcid] for srcid in self.srcids]
        jci_names_list = [jci_names[srcid] for srcid in self.srcids]


# In[ ]:

#Create a bag of words from sensor string metadata. Vectorize so that it can be used in ML algorithms.
        namevect = CountVectorizer(tokenizer=tokenizer)
        #namevect = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
        namebow = scipy.sparse.coo_matrix(namevect.fit_transform(names_list))

        descvect = CountVectorizer(tokenizer=tokenizer)
        descbow = scipy.sparse.coo_matrix(descvect.fit_transform(desc_list))

        unitvect = DictVectorizer()
        unitbow = scipy.sparse.coo_matrix(unitvect.fit_transform(unit_list))

        type_str_vect = DictVectorizer()
        type_str_bow = scipy.sparse.coo_matrix(type_str_vect.fit_transform(type_str_list))

        #typevect = DictVectorizer()
        #typebow = scipy.sparse.coo_matrix(typevect.fit_transform(type_list))

        jcivect = CountVectorizer(tokenizer=tokenizer)
        jcibow = scipy.sparse.coo_matrix(jcivect.fit_transform(jci_names_list))

        feature_set = jcivect.get_feature_names() +\
                      descvect.get_feature_names() +\
                      unitvect.get_feature_names() +\
                      type_str_vect.get_feature_names() #+\
                      #typevect.get_feature_names()

        final_bow = scipy.sparse.hstack([
                                         namebow,
                                         descbow,
                                         unitbow,
                                         type_str_bow,
                                         #typebow,
                                         jcibow
                                        ])
        self.bow_array = final_bow.toarray() # this is the bow for each sensor.

        # Hierarchical agglomerative clustering
        num_of_sensors = len(self.bow_array)
        a = np.array(self.bow_array[:num_of_sensors])
        z = linkage(a,metric='cityblock',method='complete')


        # Apply threshold to hierarchical tree to obtain individual clusters.
        # Results stored in self.cluster_map

        dists = list(set(z[:,2]))
        thresh = (dists[2] + dists[3]) /2 
        print("Threshold: ", thresh)
        b = hier.fcluster(z,thresh, criterion='distance')
        self.cluster_map = {}
        for i in range(len(b)):
            print(i, names_list[i], b[i])
            if b[i] in self.cluster_map:
                self.cluster_map[b[i]]["sensors"].append(self.sensors[i])
                self.cluster_map[b[i]]["sensor_ids"].append(i)
            else:
                self.cluster_map[b[i]] = {"sensors":[self.sensors[i]]}
                self.cluster_map[b[i]]["sensor_ids"] = [i]
            self.sensors[i]['equip_cluster_id'] = b[i]
        print('# of clusters: {0}'.format(len(self.cluster_map)))

        self.inverse_cluster_map = {}
        for cid, cluster in self.cluster_map.items():
            #for srcid in cluster['sensor_ids']:
            for sensor in cluster['sensors']:
                srcid = sensor['source_id']
                self.inverse_cluster_map[srcid] = cid

        # get ground truth set
        # Manually label say 10 clusters and hence multiple sensors. 
        self.sensor_labels = []
        self.sensor_bow = []
        self.labeled_clusters = []
        self.num_sensors_in_gray=100
        self.thresholds = [(0.1,0.95), (0.1,0.9) , (0.15,0.9), (0.15,0.85), 
                           (0.2,0.85), (0.25,0.85), (0.3,0.85), (0.35,0.85), 
                           (0.4,0.85), (0.45,0.85), (0.5,0.85), (0.55,0.85), 
                           (0.6,0.85), (0.65,0.85), (0.7,0.85), (0.75,0.85), 
                           (0.8,0.85), (0.849999999,0.85) ] 
        self.T_low = self.thresholds[0][0]
        self.T_high = self.thresholds[0][1]
        self.thresh_count = 0
        self.n_manual_lab_clusters_iter = [0]
        self.n_sensors_covered_iter = [0]
        
        """
        random_cids = random.sample(self.cluster_map.keys(), 
                                    self.seed_sample_num)
        for c_id in random_cids:
            self.labeled_clusters.append(c_id)
            for ix,i in enumerate(self.cluster_map[c_id]['sensor_ids']):
                self.sensor_bow.append(self.bow_array[i])
                source_id = self.sensors[i]['source_id']
                self.learned_srcids.append(source_id)
                #self.sensor_labels.append(self.true_sensor_types[source_id])
        #self.model.fit(self.sensor_bow, self.sensor_labels)
        """

    def get_random_learning_srcids(self, sample_num):
        srcids = []
        random_cids = random.sample(self.cluster_map.keys(), sample_num)
        for c_id in random_cids:
            #for ix,i in enumerate(self.cluster_map[c_id]['sensor_ids']):
            i = random.choice(self.cluster_map[c_id]['sensor_ids'])
            srcid = self.sensors[i]['source_id']
            srcids.append(srcid)
        return srcids

    def update_model(self, new_srcids, new_labels):
        for srcid, new_label in zip(new_srcids, new_labels):
            if not new_label:
                raise Exception('Point Tagset not found at {0}'.format(srcid))
            self.learned_srcids.append(srcid)
            i = self.srcids.index(srcid)
            self.sensor_bow.append(self.bow_array[i])
            self.sensor_labels.append(new_label)
            cid = self.inverse_cluster_map[srcid]
            self.labeled_clusters.append(cid)
            cluster = self.cluster_map[cid]
            sensor_ids = cluster['sensor_ids']
            for sensor_id in sensor_ids:
                self.sensor_bow.append(self.bow_array[sensor_id])
                self.sensor_labels.append(new_label)
        self.model.fit(self.sensor_bow, self.sensor_labels)

        # Iteratively apply Random Forest to label new sensors
        self.change_thresholds = True
        n_wrong_confident_sensor_pred = 0
        self.sensor_bow = list(self.sensor_bow)
        self.sensor_labels = list(self.sensor_labels)
        n_high_confidence_sensors = 0
        confidence_dict = {}
        for p in self.cluster_map.keys(): # for each cluster
            if p in self.labeled_clusters:
                continue
            # Get sensors from this cluster.
            sample_bow = []
            for k in self.cluster_map[p]['sensor_ids']:
                sample_bow.append(self.bow_array[k])
            sample_bow = np.array(sample_bow)

            # Apply trained model:
            confidence = self.model.predict_proba(sample_bow)
            prediction_label = self.model.predict(sample_bow)
            # Get overall max confidence for any sensor in cluster:
            confidence_dict[p] = confidence
            max_c = 0
            for c in confidence:
                max_c = max(np.append(c,[max_c]))

            # Compare with Thresholds.
            flag = 0
            if max_c < self.T_low:
                flag = 1
            if max_c > self.T_high:
                flag = 2


            # Handle the cluster beyond threshold:
            if flag > 0:
                self.change_thresholds = False
                self.labeled_clusters.append(p)

                # For each sensor in this cluster:
                for k in range(len(self.cluster_map[p]['sensors'])):
                    sourceid = self.cluster_map[p]['sensors'][k]['source_id']
                    #true_type = self.true_sensor_types[sourceid]
                    pred_type = prediction_label[k]

                    if flag==2:
                        n_high_confidence_sensors += 1
                        #if pred_type != true_type:
                        #    n_wrong_confident_sensor_pred+=1

                        # append these sensors into labeled ones (with possibly wrong labels):
                        self.sensor_bow.append(
                            self.bow_array[self.cluster_map[p]['sensor_ids'][k]])
                        self.sensor_labels.append(pred_type)
                break
        self.model.fit(self.sensor_bow, self.sensor_labels)
        if self.change_thresholds:
            self.thresh_count += 1
        self.T_low, self.T_high = self.thresholds[self.thresh_count]
        print(self.T_low, self.T_high)

    def learn_one_step(self, todo_sample_num=10):
        added_sample_cnt = 0
        while added_sample_cnt < todo_sample_num and \
              self.num_sensors_in_gray > 0:
            self.T_low, self.T_high = self.thresholds[self.thresh_count]
            self.model.fit(self.sensor_bow, self.sensor_labels)

            # Use model to label clusters/sensors:
            n_manually_labeled_thisepoch, \
            n_wrong_confident_sensor_pred, \
            n_high_confidence_sensors, \
            n_sens_covered, \
            perc_coverage,\
            self.change_thresholds,\
            self.num_sensors_in_gray,\
            new_sample_cnt          = self.select_informative_samples(
                                          todo_sample_num - added_sample_cnt)
            print(n_manually_labeled_thisepoch, \
                  n_wrong_confident_sensor_pred, \
                  n_high_confidence_sensors, \
                  n_sens_covered, \
                  perc_coverage,\
                  self.change_thresholds,\
                  self.num_sensors_in_gray)
            added_sample_cnt += new_sample_cnt
            self.n_manual_lab_clusters_iter.append(
                self.n_manual_lab_clusters_iter[-1] + 
                n_manually_labeled_thisepoch)
            self.n_sensors_covered_iter.append(n_sens_covered)
            if self.change_thresholds:
                self.thresh_count += 1
                print('New thresholds: < {0} || > {1}'\
                          .format(self.T_low, self.T_high))
        print('# of newly added samples: {0}'.format(added_sample_cnt))
        return self.num_sensors_in_gray

    def get_num_sensors_in_gray(self):
        return len(self.bow_array) - len(self.sensor_labels)
        

    def learn_step_by_step(self):
        
        #self.sensor_bow = list(self.sensor_bow)
        #self.sensor_labels = list(self.sensor_labels)
        random_cids = random.sample(self.cluster_map.keys(), 
                                    self.seed_sample_num)
        for c_id in random_cids:
            self.labeled_clusters.append(c_id)
            for ix,i in enumerate(self.cluster_map[c_id]['sensor_ids']):
                self.sensor_bow.append(self.bow_array[i])
                source_id = self.sensors[i]['source_id']
                self.learned_srcids.append(source_id)
                self.sensor_labels.append(self.true_sensor_types[source_id])
        #self.sensor_bow = np.array(self.sensor_bow)
        #self.sensor_labels = np.array(self.sensor_labels)

        while self.num_sensors_in_gray > 0:
            self.num_sensors_in_gray = self.learn_one_step(10)
            print('manually labeled: {0}'\
                      .format(self.n_manual_lab_clusters_iter))
        
        print('# of manual samples: {0}'
                  .format(self.n_manual_lab_clusters_iter))

    def plot_result(self):
        fig, plot_list = plotter.plot_multiple_2dline(
                             self.n_manual_lab_clusters_iter, 
                             [self.n_sensors_covered_iter])
        plotter.save_fig(fig, 'exp-1.pdf')

    def baseline(self):
        desc_list=[]
        jc_names_list=[]
        sensor_info = {}
        for s in self.sensors:
            sid = s['source_id']
            sensor_info[sid]={}
            d = s['description'].lower()
            d = ''.join([i for i in d if not i.isdigit()]) #remove digits
            d = re.sub(r"[^\w' ]", "",  d ) # remove special chars
            d = ' '.join(d.split()) #remove extra spaces
            sensor_info[sid]['desc'] = d
            desc_list.append(d)

            j = s['jci_name'].split('.')[-1]
            sensor_info[sid]['jci'] = j
            jc_names_list.append(j)
            sensor_info[sid]['figuredout'] = False

        manualeffort=[0]
        coveredsensors=[0]
        desc_map = {}
        jci_map = {}

        for s in self.sensors:
            sid = s['source_id']
            if sensor_info[sid]['figuredout']==True: continue # If label known, skip.

            # Info about this sensor:
            gt = self.true_sensor_types[ s['source_id'] ]  # ground truth
            d = s['description'].lower()
            d = ''.join([i for i in d if not i.isdigit()]) #remove digits
            d = re.sub(r"[^\w' ]", "",  d ) # remove special chars
            d = ' '.join(d.split()) #remove extra spaces
            j = s['jci_name'].split('.')[-1]

            if d:
                if not d in desc_map:
                    manualeffort.append(manualeffort[-1]+1)
                    desc_map[d] = gt
                    jci_map[j] = gt
                    sensor_info[sid]['figuredout']=True
                    # Check how many it catches:
                    numcatches=0
                    for s2 in self.sensors:
                        sid2 = s2['source_id']
                        if sensor_info[sid2]['figuredout']==False and \
                               (sensor_info[sid2]['desc']==d or \
                               sensor_info[sid2]['jci']==j):
                            sensor_info[sid2]['figuredout']=True
                            numcatches+=1
                    coveredsensors.append(coveredsensors[-1] + numcatches)
            else:
                if not j in jci_map:
                    manualeffort.append(manualeffort[-1]+1)
                    jci_map[j] = gt
                    sensor_info[sid]['figuredout']=True
                    # Check how many it catches:
                    numcatches=0
                    for s2 in self.sensors:
                        sid2 = s2['source_id']
                        if sensor_info[sid2]['figuredout'] == False and \
                            sensor_info[sid2]['jci'] == j:
                            sensor_info[sid2]['figuredout'] = True
                            numcatches+=1
                    coveredsensors.append(coveredsensors[-1] + numcatches)

    # Just checking.
        for s in self.sensors:
            sid = s['source_id']
            if sensor_info[sid]['figuredout']==False:
                print(sid)
        
        fig, plot_list = plotter.plot_multiple_2dline(manualeffort, 
                                                      [coveredsensors])
        plotter.save_fig(fig, 'test-2.pdf')
        print('done printing')

    def learn_to_end(self):
        self.sensor_bow = list(self.sensor_bow)
        self.sensor_labels = list(self.sensor_labels)
        random_cids = random.sample(self.cluster_map.keys(), 
                                    self.seed_sample_num)
        for c_id in random_cids:
            self.labeled_clusters.append(c_id)
            for ix,i in enumerate(self.cluster_map[c_id]['sensor_ids']):
                self.sensor_bow.append(self.bow_array[i])
                source_id = self.sensors[i]['source_id']
                self.learned_srcids.append(source_id)
                self.sensor_labels.append(self.true_sensor_types[source_id])
        self.sensor_bow = np.array(self.sensor_bow)
        self.sensor_labels = np.array(self.sensor_labels)

        while self.num_sensors_in_gray > 0:
            self.model.fit(self.sensor_bow, self.sensor_labels)
            self.T_low, self.T_high = self.thresholds[self.thresh_count]

            # Use model to label clusters/sensors:
            n_manually_labeled_thisepoch, \
            n_wrong_confident_sensor_pred, \
            n_high_confidence_sensors, \
            n_sens_covered, \
            perc_coverage,\
            self.change_thresholds,\
            self.num_sensors_in_gray = self.apply_model_on_all_clusters()
            print(n_manually_labeled_thisepoch, 
                  n_wrong_confident_sensor_pred, 
                  n_high_confidence_sensors, 
                  n_sens_covered, 
                  perc_coverage,
                  self.change_thresholds, 
                  self.num_sensors_in_gray)
            self.n_manual_lab_clusters_iter.append(
                self.n_manual_lab_clusters_iter[-1] + 
                n_manually_labeled_thisepoch)
            self.n_sensors_covered_iter.append(n_sens_covered)

            if self.change_thresholds:
                self.thresh_count += 1
                print(self.T_low, self.T_high)

            # Re-train model:


        # This code uses regular expressions to map descriptions (and if needed, jci_name) to ground truth
        # Goal: Get a manual effort (in mapping either of above to ground truth) to coverage
        print('# of manual samples: {0}'
                  .format(self.n_manual_lab_clusters_iter))

        desc_list=[]
        jc_names_list=[]
        sensor_info = {}
        for s in self.sensors:
            sid = s['source_id']
            sensor_info[sid]={}
            d = s['description'].lower()
            d = ''.join([i for i in d if not i.isdigit()]) #remove digits
            d = re.sub(r"[^\w' ]", "",  d ) # remove special chars
            d = ' '.join(d.split()) #remove extra spaces
            sensor_info[sid]['desc'] = d
            desc_list.append(d)

            j = s['jci_name'].split('.')[-1]
            sensor_info[sid]['jci'] = j
            jc_names_list.append(j)
            sensor_info[sid]['figuredout'] = False

        manualeffort=[0]
        coveredsensors=[0]
        desc_map = {}
        jci_map = {}

        for s in self.sensors:
            sid = s['source_id']
            if sensor_info[sid]['figuredout']==True: continue # If label known, skip.

            # Info about this sensor:
            gt = self.true_sensor_types[ s['source_id'] ]  # ground truth
            d = s['description'].lower()
            d = ''.join([i for i in d if not i.isdigit()]) #remove digits
            d = re.sub(r"[^\w' ]", "",  d ) # remove special chars
            d = ' '.join(d.split()) #remove extra spaces
            j = s['jci_name'].split('.')[-1]

            if not d =="":
                if not d in desc_map:
                    manualeffort.append(manualeffort[-1]+1)
                    desc_map[d] = gt
                    jci_map[j] = gt
                    sensor_info[sid]['figuredout']=True
                    # Check how many it catches:
                    numcatches=0
                    for s2 in self.sensors:
                        sid2 = s2['source_id']
                        if sensor_info[sid2]['figuredout']==False and \
                            (sensor_info[sid2]['desc']==d or \
                            sensor_info[sid2]['jci']==j):
                            sensor_info[sid2]['figuredout']=True
                            numcatches+=1
                    coveredsensors.append(coveredsensors[-1] + numcatches)
            else:
                if not j in jci_map:
                    manualeffort.append(manualeffort[-1]+1)
                    jci_map[j] = gt
                    sensor_info[sid]['figuredout']=True
                    # Check how many it catches:
                    numcatches=0
                    for s2 in self.sensors:
                        sid2 = s2['source_id']
                        if sensor_info[sid2]['figuredout']==False and \
                            sensor_info[sid2]['jci']==j:
                            sensor_info[sid2]['figuredout']=True
                            numcatches+=1
                    coveredsensors.append(coveredsensors[-1] + numcatches)

    # Just checking.
        for s in self.sensors:
            sid = s['source_id']
            if sensor_info[sid]['figuredout']==False:
                print(sid)


    # Plot the manual effort vs coverage for Regex based approach.
        fig, plot_list = plotter.plot_multiple_2dline(manualeffort, 
                                                      [coveredsensors])
        plotter.save_fig(fig, 'test-2.pdf')

        len(set(jc_names_list))

    def apply_model_on_all_clusters(self): # model, T_low, T_high, ....
        # This method uses global variables including model, T_low, T_high and several others.
        # Goal: apply model on all clusters and determine correctness of "confident predictions"
        # and manually label "very low confidence" ones

        # Iteratively apply Random Forest to label new sensors
        self.change_thresholds = True
        n_wrong_confident_sensor_pred = 0
        self.sensor_bow = list(self.sensor_bow)
        self.sensor_labels = list(self.sensor_labels)
        n_high_confidence_sensors = 0
        n_manually_labeled_thisepoch = 0 # epoch = whatever happens after (re-) training RF models

        for p in self.cluster_map.keys(): # for each cluster
        #for p in sorted_equip_keys:
            #p = p[0]

            # Escape if already labeled.
            if p in self.labeled_clusters:
                continue


            # Get sensors from this cluster.
            sample_bow = []
            for k in self.cluster_map[p]['sensor_ids']:
                sample_bow.append(self.bow_array[k])
            sample_bow = np.array(sample_bow)


            # Apply trained model:
            confidence = self.model.predict_proba(sample_bow)
            prediction_label = self.model.predict(sample_bow)
            # Get overall max confidence for any sensor in cluster:
            max_c = 0
            for c in confidence:
                max_c = max(np.append(c,[max_c]))


            # Compare with Thresholds.
            flag = 0
            if max_c < self.T_low:
                flag = 1
            if max_c > self.T_high:
                flag = 2

            if flag==1:
                n_manually_labeled_thisepoch+=1


            # Handle the cluster beyond threshold:
            if flag > 0:
                self.change_thresholds = False
                self.labeled_clusters.append(p)

                # For each sensor in this cluster:
                for k in range(len(self.cluster_map[p]['sensors'])):
                    sourceid = self.cluster_map[p]['sensors'][k]['source_id']
                    true_type = self.true_sensor_types[sourceid]
                    pred_type = prediction_label[k]

                    if flag == 2:
                        n_high_confidence_sensors+=1
                        if pred_type != true_type:
                            n_wrong_confident_sensor_pred+=1

                        # append these sensors into labeled ones (with possibly wrong labels):
                        self.sensor_bow.append(self.bow_array[
                            self.cluster_map[p]['sensor_ids'][k]])
                        self.sensor_labels.append(pred_type)

                    if flag == 1:
                        # append these sensors into labeled ones (with ground truth):
                        self.sensor_bow.append(self.bow_array[
                            self.cluster_map[p]['sensor_ids'][k]])
                        self.sensor_labels.append(true_type)
                break

        return n_manually_labeled_thisepoch, \
               n_wrong_confident_sensor_pred, \
               n_high_confidence_sensors, \
               len(self.sensor_labels), \
               100.0*len(self.sensor_labels)/len(self.bow_array), \
               self.change_thresholds, \
               len(self.bow_array)-len(self.sensor_labels)

    def predict(self, target_srcids):
        target_bow = [self.bow_array[self.srcids.index(srcid)] 
                      for srcid in target_srcids]
        pred = self.model.predict(target_bow)
        return pred
    
    def predict_proba(self, target_srcids):
        target_bow = [self.bow_array[self.srcids.index(srcid)] 
                      for srcid in target_srcids]
        proba = self.model.predict_proba(target_bow)
        return proba


def main():

    with open('metadata/bacnet_devices.json', 'r') as fp:
        sensors_dict = json.load(fp)

    device_list = [
                    "557",
                    "607",
                    "608",
                    "609",
                    "610",
    ]

    names = dict()
    descs = dict()
    units = dict()
    type_strs = dict()
    types = dict()
    jci_names = dict()
    source_ids = set([])
    for nae in device_list:
        device = sensors_dict[nae]
        h_dev = device['props']
        for sensor in device['objs']:
            h_obj = sensor['props']
            srcid = str(h_dev['device_id']) + '_' + \
                    str(h_obj['type']) + '_' + \
                    str(h_obj['instance'])

            if h_obj['type'] not in (0,1,2,3,4,5,13,14,19):
                continue

            if srcid in source_ids:
                continue
            else:
                source_ids.add(srcid)

            # create individual lists
            # remove numbers from names 
            # because they do not indicate type of sensor
            names[srcid] = ''.join([c for c in sensor['name'] 
                                    if not c.isdigit()])
            descs[srcid] = ''.join([c for c in sensor['desc'] 
                                    if not c.isdigit()])
            jci_names[srcid] = ''.join([c for c in sensor['jci_name'] 
                                        if not c.isdigit()])
            units[srcid] = {str(sensor['unit']):1}
            type_strs[srcid] = {str(h_obj['type_str']):1}
            types[srcid] = {str(h_obj['type']):1}

    true_df = pd.read_csv('metadata/bonner_sensor_types.csv')
    true_sensor_types = {s[1]['source_id']:s[1]['sensor_type'] 
                         for s in true_df.iterrows()}
    zodiac = Zodiac(names, descs, units, type_strs, types, jci_names, 
                    true_sensor_types)
    zodiac.learn_step_by_step()
    zodiac.plot_result()
    zodiac.baseline()


if __name__ == '__main__':
    main()
