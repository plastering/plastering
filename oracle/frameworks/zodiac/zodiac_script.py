import numpy as np
import pandas as pd
from scipy.cluster.vq import *
import operator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle as pkl
import shelve
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from sklearn.feature_extraction import DictVectorizer
import json
import pdb
import arrow

import plotter

#Import shelve dictionary containing all bacnet devices
#sensors_dict = shelve.open('metadata/bacnet_devices.shelve','r')
with open('metadata/bacnet_devices.json', 'r') as fp:
    sensors_dict = json.load(fp)

pdb.set_trace()


#bonner hall
device_list = [
                "557",
                "607",
                "608",
                "609",
                "610",
]


#Gives us a sensor_list with sensor information of a building
sensor_list = []
names_list = []
names_listWithDigits = [] 
sensor_type_namez=[]
desc_list = []
unit_list = []
type_str_list = []
type_list = []
jci_names_list = []
source_id_set = set([])
for nae in device_list:
    device = sensors_dict[nae]
    h_dev = device['props']
    for sensor in device['objs']:
        h_obj = sensor['props']
        source_id = str(h_dev['device_id']) + '_' + str(h_obj['type']) + '_' + str(h_obj['instance'])

        if h_obj['type'] not in (0,1,2,3,4,5,13,14,19):
            continue

        if source_id in source_id_set:
            continue
        else:
            source_id_set.add(source_id)

        #create individual lists
        #remove numbers from names because they do not indicate type of sensor
        names_listWithDigits.append(sensor['jci_name']) 
        sensor_type_namez.append(sensor['sensor_type'])
        names_list.append(''.join([c for c in sensor['name'] if not c.isdigit()]))
        desc_list.append(''.join([c for c in sensor['desc'] if not c.isdigit()]))
        jci_names_list.append(''.join([c for c in sensor['jci_name'] if not c.isdigit()]))
        #convert string to dictionary for categorical vectorization
        unit_list.append({str(sensor['unit']):1})
        type_str_list.append({str(h_obj['type_str']):1})
        type_list.append({str(h_obj['type']):1})

        #create a flat list of dictionary to avoid using json file
        sensor_list.append({'source_id': source_id,
                            'name': sensor['name'],
                            'description': sensor['desc'],
                            'unit': sensor['unit'],
                            'type_string': h_obj['type_str'],
                            'type': h_obj['type'],
                            #'device_id': h_obj['device_id'],
                            'jci_name': sensor['jci_name'],
                            #add data related characteristics here
                        })
sensor_df = pd.DataFrame(sensor_list)
sensor_df = sensor_df.set_index('source_id')
sensor_df = sensor_df.groupby(sensor_df.index).first()
print(len(sensor_list))


# In[ ]:

#Create a bag of words from sensor string metadata. Vectorize so that it can be used in ML algorithms.
namevect = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
namebow = scipy.sparse.coo_matrix(namevect.fit_transform(names_list))

descvect = CountVectorizer() 
descbow = scipy.sparse.coo_matrix(descvect.fit_transform(desc_list))

unitvect = DictVectorizer() 
unitbow = scipy.sparse.coo_matrix(unitvect.fit_transform(unit_list))

type_str_vect = DictVectorizer() 
type_str_bow = scipy.sparse.coo_matrix(type_str_vect.fit_transform(type_str_list))

typevect = DictVectorizer() 
typebow = scipy.sparse.coo_matrix(typevect.fit_transform(type_list))

jcivect = CountVectorizer() 
jcibow = scipy.sparse.coo_matrix(jcivect.fit_transform(jci_names_list))

feature_set = jcivect.get_feature_names()+               descvect.get_feature_names()+               unitvect.get_feature_names()+               type_str_vect.get_feature_names()+               typevect.get_feature_names()
              

final_bow = scipy.sparse.hstack([
                                 #namebow,
                                 descbow,
                                 unitbow,
                                 type_str_bow,
                                 typebow,
                                 jcibow
                                ]) 
bow_array = final_bow.toarray() # this is the bow for each sensor. 


# In[ ]:

# Hierarchical agglomerative clustering 
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as hier

num_of_sensors = len(bow_array)
a = np.array(bow_array[:num_of_sensors])
z = linkage(a,metric='cityblock',method='complete')


# In[ ]:

#Apply threshold to hierarchical tree to obtain individual clusters. Results stored in equip_map
dists = list(set(z[:,2]))
thresh = (dists[2] + dists[3]) /2 
print("Threshold: ", thresh)
b = hier.fcluster(z,thresh, criterion='distance')
cluster_map = {}
equip_map = {}
for i in range(len(b)):
    cluster_map[names_list[i]] = b[i]
    print(i, names_list[i], b[i])
    if b[i] in equip_map:
        equip_map[b[i]]["sensors"].append(sensor_list[i])
        equip_map[b[i]]["sensor_ids"].append(i)
    else:
        equip_map[b[i]] = {"sensors":[sensor_list[i]]}
        equip_map[b[i]]["sensor_ids"] = [i]
    sensor_list[i]['equip_cluster_id'] = b[i]
sorted_map = sorted(cluster_map.items(), key=operator.itemgetter(1))


# In[ ]:

#read ground truth sensor types
import csv
building = 'bonner'
ground_truth_list = []
with open('metadata/'+building+'_sensor_types.csv') as ground_truth_file:
    csv_reader = csv.DictReader(ground_truth_file)
    for row in csv_reader:
        ground_truth_list.append(row)
sensor_type_map = {s['source_id']:s['sensor_type'] for s in ground_truth_list}


# In[ ]:

# Merges the clusters formed by hierarchical clustering based on "description" tag. 
equip_desc_map = {}
sensor_abbrvs = [s['jci_name'].split('.')[-1].lower() if '.' in s['jci_name'] else s['jci_name'] for s in ground_truth_list]
#sensor_abbrvs = [re.sub('[^a-z ]', '', s) for s in sensor_abbrvs]

for k,v in equip_map.items():
    #print v
    desc_list = [s['description'].lower() for s in v['sensors']]
    desc_list = [re.sub('[^a-z ]', '', d) for d in desc_list]
    desc_list = [sensor_abbrvs[i] if d == '' else d for i,d in enumerate(desc_list)]
    if len(set(desc_list)) == 1:
        if desc_list[0] in equip_desc_map and desc_list[0] != '':
            equip_desc_map[desc_list[0]]['sensors'] += v['sensors']
            equip_desc_map[desc_list[0]]['sensor_ids'] += v['sensor_ids']
        elif desc_list[0] == '':
            equip_desc_map[k] = v
        else:
            equip_desc_map[desc_list[0]] = v
    else:
        equip_desc_map[k] = v
    
#print "merged cluster:", len(equip_desc_map)


# In[ ]:

#get ground truth set
#equip_map = equip_desc_map #Uncomment for using merged clusters
# Manually label say 10 clusters and hence multiple sensors. 
import random
num_manual_labels = 10
sensor_labels = []
sensor_bow = []
labeled_equip_keys = []
equip_cluster_lens = {k:len(v['sensors']) for k,v in equip_map.items()}
sorted_equip_keys = sorted(equip_cluster_lens.items(), key=operator.itemgetter(1), reverse=True)
#random_cids = random.sample(equip_map.keys(), num_manual_labels)
random_cids = [241, 112, 291, 99, 236, 1, 68, 124, 229, 208]
cnt = 0
for c_id in random_cids:
    labeled_equip_keys.append(c_id)
    for ix,i in enumerate(equip_map[c_id]['sensor_ids']):
            cnt += 1
            sensor_bow.append(bow_array[i])
            source_id = sensor_list[i]['source_id']
            sensor_labels.append(sensor_type_map[source_id])
print(cnt)
print('============')
for cid in random_cids:
    print(cid, len(equip_map[cid]['sensor_ids']))
print('============')
print(sum([len(c['sensor_ids']) for c in equip_map.values()]))
print('============')
sensor_bow = np.array(sensor_bow)
sensor_labels = np.array(sensor_labels)


# In[ ]:

#learn a model
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
le = LabelEncoder()
le.fit(sensor_labels)
#print list(le.classes_)
train_labels = le.transform(sensor_labels)
model = RandomForestClassifier(n_estimators=400, random_state=0)
model.fit(sensor_bow,sensor_labels)
#model.fit(sensor_bow)


# In[ ]:

DEBUG = True

def apply_model_on_all_clusters( ): # model, T_low, T_high, ....
    # This method uses global variables including model, T_low, T_high and several others. 
    # Goal: apply model on all clusters and determine correctness of "confident predictions" 
    # and manually label "very low confidence" ones 

    global sensor_labels
    global sensor_bow 
    global labeled_equip_keys 
    
    # Iteratively apply Random Forest to label new sensors 
    change_thresholds = True 
    n_wrong_confident_sensor_pred = 0
    sensor_bow = list(sensor_bow)
    sensor_labels = list(sensor_labels) 
    n_high_confidence_sensors = 0
    n_manually_labeled_thisepoch = 0 # epoch = whatever happens after (re-) training RF models

    for p in equip_map.keys(): # for each cluster 
    #for p in sorted_equip_keys:
        #p = p[0]

        # Escape if already labeled. 
        if p in labeled_equip_keys:
            continue


        # Get sensors from this cluster. 
        sample_bow = []
        for k in equip_map[p]['sensor_ids']:
            sample_bow.append(bow_array[k])
        sample_bow = np.array(sample_bow)


        # Apply trained model: 
        confidence = model.predict_proba(sample_bow)
        prediction_label = model.predict(sample_bow)
        # Get overall max confidence for any sensor in cluster: 
        max_c = 0
        for c in confidence:
            max_c = max(np.append(c,[max_c]))


        # Compare with Thresholds. 
        flag = 0    
        if max_c < T_low:
            flag = 1
        if max_c > T_high:
            flag = 2        

        if flag==1: 
            n_manually_labeled_thisepoch+=1 

        
        # Handle the cluster beyond threshold: 
        if flag>0: 
            change_thresholds = False 
            labeled_equip_keys.append(p)  

            # For each sensor in this cluster: 
            for k in range(len(equip_map[p]['sensors'])):  
                sourceid = equip_map[p]['sensors'][k]['source_id']
                true_type = sensor_type_map[sourceid] 
                pred_type = prediction_label[k]              

                if flag==2: 
                    n_high_confidence_sensors+=1
                    if pred_type != true_type: 
                        n_wrong_confident_sensor_pred+=1 

                    # append these sensors into labeled ones (with possibly wrong labels): 
                    sensor_bow.append(bow_array[equip_map[p]['sensor_ids'][k]]) 
                    sensor_labels.append(pred_type) 

                if flag==1:                 
                    
                    # append these sensors into labeled ones (with ground truth): 
                    sensor_bow.append(bow_array[equip_map[p]['sensor_ids'][k]]) 
                    sensor_labels.append(true_type) 

            break
        #sensor_bow = np.array(sensor_bow)
        #sensor_labels = np.array(sensor_labels)
        #model.fit(sensor_bow, sensor_labels)
        
    return n_manually_labeled_thisepoch, n_wrong_confident_sensor_pred, n_high_confidence_sensors, len(sensor_labels), 100.0*len(sensor_labels)/len(bow_array),change_thresholds,len(bow_array)-len(sensor_labels)   
    # return __  , __ , __ , num labeled sensors , % labeled sensors, change_thresholds, num_sensors_in_gray 


# In[ ]:

# Iteratively train RF model and call the method to apply it on all clusters. 
# When method asks us to change thresholds, then we do so. 
# Otherwise we re-train RF and try catch more sensors. 
# We also record the number of correct sensors in each iteration, the manual effort in each iteration etc. 

#print equip_map.keys() 
#print labeled_equip_keys 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=400, random_state=0)
model.fit(sensor_bow,sensor_labels) 

num_sensors_in_gray=100

T_low = 0.1
T_high = 0.95 
thresholds = [ (0.1,0.95), (0.1,0.9) , (0.15,0.9), (0.15,0.85), (0.2,0.85), (0.25,0.85), (0.3,0.85), (0.35,0.85), (0.4,0.85), (0.45,0.85), (0.5,0.85), (0.55,0.85), (0.6,0.85), (0.65,0.85), (0.7,0.85), (0.75,0.85), (0.8,0.85), (0.849999999,0.85) ] 
#thresholds = [ (0.1,0.9) , (0.15,0.9), (0.15,0.85), (0.2,0.85), (0.25,0.85), (0.3,0.85), (0.35,0.85), (0.4,0.85), (0.45,0.85), (0.45,0.8),(0.5,0.8), (0.55,0.8), (0.6,0.8), (0.65,0.8), (0.7,0.8), (0.75,0.8), (0.7999999,0.8) ] 
#thresholds = [ (0.1,0.7) , (0.25,0.7), (0.3,0.7), (0.3,0.7), (0.35,0.7), (0.4,0.7), (0.4,0.65),(0.45,0.65), (0.5,0.65), (0.5,0.6), (0.55,0.6), (0.5999999,0.6)] 

thresh_count=0 

# Start iterations: 
n_manual_lab_clusters_iter = [10 ]
n_sensors_covered_iter = [len(sensor_labels) ] 

while num_sensors_in_gray>0: 
    T_low,T_high = thresholds[thresh_count] 
    
    # Re-train model: 
    model.fit(sensor_bow,sensor_labels) 
    
    # Use model to label clusters/sensors: 
    n_manually_labeled_thisepoch, n_wrong_confident_sensor_pred, n_high_confidence_sensors, n_sens_covered, perc_coverage,change_thresholds,num_sensors_in_gray = apply_model_on_all_clusters()        
    print(n_manually_labeled_thisepoch, n_wrong_confident_sensor_pred, n_high_confidence_sensors, n_sens_covered, perc_coverage,change_thresholds,num_sensors_in_gray)
    n_manual_lab_clusters_iter.append(n_manual_lab_clusters_iter[-1] + n_manually_labeled_thisepoch) 
    n_sensors_covered_iter.append(n_sens_covered) 
    
    if change_thresholds: 
        thresh_count+=1 
        print('new thresholds: ', T_low, T_high)


# In[ ]:

# This code uses regular expressions to map descriptions (and if needed, jci_name) to ground truth 
# Goal: Get a manual effort (in mapping either of above to ground truth) to coverage 

desc_list=[] 
jc_names_list=[] 
sensor_info = {} 
for s in sensor_list: 
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

for s in sensor_list: 
    sid = s['source_id'] 
    if sensor_info[sid]['figuredout']==True: continue # If label known, skip. 
        
    # Info about this sensor: 
    gt = sensor_type_map[ s['source_id'] ]  # ground truth 
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
            for s2 in sensor_list: 
                sid2 = s2['source_id'] 
                if sensor_info[sid2]['figuredout']==False and (sensor_info[sid2]['desc']==d   or sensor_info[sid2]['jci']==j): 
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
            for s2 in sensor_list: 
                sid2 = s2['source_id'] 
                if sensor_info[sid2]['figuredout']==False and sensor_info[sid2]['jci']==j: 
                    sensor_info[sid2]['figuredout']=True
                    numcatches+=1 
            coveredsensors.append(coveredsensors[-1] + numcatches) 
            
            
            
    
# Just checking. 
for s in sensor_list: 
    sid = s['source_id'] 
    if sensor_info[sid]['figuredout']==False: 
        print(sid )


        

# Plot the manual effort vs coverage for Regex based approach. 
fig, plot_list = plotter.plot_multiple_2dline(manualeffort, [coveredsensors])
plotter.save_fig(fig, 'test-script1.pdf')

fig, plot_list = plotter.plot_multiple_2dline(n_manual_lab_clusters_iter, 
                                              [n_sensors_covered_iter])
plotter.save_fig(fig, 'exp-script.pdf')
"""
plt.plot(manualeffort, coveredsensors, 'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.ylabel('# points covered', fontsize=20)
plt.xlabel('Manual inputs', fontsize=20)
plt.tight_layout()
plt.xticks(np.arange(0, max(manualeffort)+1, 75.))
plt.savefig("BonnersensorsREGEXManualVsCoverage.pdf",bbox_inches='tight',dpi=150)
"""


len(set(jc_names_list))

