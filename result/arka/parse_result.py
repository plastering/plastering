import os
import sys
import pdb
import re
from copy import deepcopy
from operator import itemgetter
import json

import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/../..')
from plastering.metadata_interface import *
from plastering.evaluator import *

target_building = 'sdh'
currfile = __file__
base_dir = os.path.dirname(currfile)
target_dir = base_dir + '/' + target_building


def get_number(s):
    return int(re.findall('\\d+', s)[0])

def is_finished():
    for cid, curr_eid in curr_eids.items():
        if curr_eid < len(qualified_examples_nums[cid]) - 1:
            return False
    return True

def select_next_cid():
    ordered_cids = [row[0] for row in
                    sorted(curr_cluster_sizes.items(),
                           key=itemgetter(1),
                           reverse=True)]
    for cid in ordered_cids:
        curr_eid = curr_eids[cid]
        if curr_eid < len(qualified_examples_nums[cid]) - 1:
            return cid
    raise Exception('cannot find cids without finishing the algorithm. A bug')

def get_srcid(name):
    return '_'.join(re.findall('[a-zA-Z0-9]+', name))

orig_cluster_sizes = {}
total_names = []
for filename in os.listdir(target_dir):
    if not re.match('{0}-ORIGINAL-METADATA-\\d+$'.format(target_building.upper()),
                    filename):
        continue
    cid = get_number(filename)
    with open(target_dir + '/' + filename, 'r') as fp:
        names = fp.readlines()
    orig_cluster_sizes[cid] = len(names)
    total_names += names
total_names = list(set(total_names))
total_srcids = [get_srcid(name) for name in total_names]
curr_cluster_sizes = deepcopy(orig_cluster_sizes)

true_tagsets = {srcid: LabeledMetadata.objects(srcid=srcid).first().tagsets
                for srcid in total_srcids}
true_points = {srcid: LabeledMetadata.objects(srcid=srcid).first().point_tagset
                for srcid in total_srcids}

qualified_examples_nums = {}
for filename in os.listdir(target_dir):
    if not re.match('l-ex-\\d+-out$', filename):
        continue
    cid = get_number(filename)
    df = pd.read_csv(target_dir + '/' + filename)
    df.columns = df.columns.str.strip()
    coverages = df['Num Examples Thought to be fully qualified'].tolist()
    qualified_examples_nums[cid] = coverages


inferred_points_dict = {i: {} for i in curr_cluster_sizes.keys()}
for filename in os.listdir(target_dir):
    if not re.match('l-ex-\\d+-out-points-qualified$', filename):
        continue
    cid = get_number(filename)
    with open(target_dir + '/' + filename, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        ex_id = int(line.split(' ')[0])
        if "'" not in line:
            items = []
        else:
            items = line.split('[')[-1].split(']')[0][1:-1].split("', '")
        inferred_points_dict[cid][ex_id] = items

pred = {}

curr_eids = {i: 0 for i in curr_cluster_sizes.keys()}


total_num = sum(orig_cluster_sizes.values())

pred_names = set()
cnt = 0
accs = []
f1s = []
mf1s = []
anymf1s = []
srcids = []
pred = {srcid: [] for srcid in total_srcids}
point_pred = {srcid: [] for srcid in total_srcids}
res = []

while not is_finished():
    # select cluster
    #max_cid = max(curr_cluster_sizes.items(), key=itemgetter(1))[0]
    cnt += 1
    max_cid = select_next_cid()
    curr_eids[max_cid] += 1
    curr_eid = curr_eids[max_cid]
    found_names = set(inferred_points_dict[max_cid][curr_eid])
    new_names = found_names - pred_names
    new_srcids = [get_srcid(name) for name in new_names]
    pred_names = pred_names.union(new_names)
    curr_cluster_sizes[max_cid] = orig_cluster_sizes[max_cid] - len(found_names)
    acc = len(pred_names) / total_num
    print('{0}\tacc: {1}'.format(cnt, acc))
    pred.update({srcid: LabeledMetadata.objects(srcid=srcid).first().tagsets
                 for srcid in new_srcids})
    point_pred.update({
        srcid: LabeledMetadata.objects(srcid=srcid).first().point_tagset
        for srcid in new_srcids})
    anymf1 = get_macro_f1(true_tagsets, pred)
    mf1 = get_macro_f1(true_points, point_pred)
    f1 = get_micro_f1(true_points, point_pred)
    #mf1s.append(mf1)
    #f1s.append(f1)
    #anymf1s.append(anymf1)
    #accs.append(acc)
    #srcids.append(len(pred_names))
    row = {
        'metrics': {
            'f1': f1,
            'macrof1': mf1,
            'accuracy': acc,
            'macrof1-all': anymf1
        },
        'learning_srcids': cnt
    }
    res.append(row)


with open('result/pointonly_notransfer_arka_{0}_0.json'.format(target_building),
          'w') as fp:
    json.dump(res, fp)
