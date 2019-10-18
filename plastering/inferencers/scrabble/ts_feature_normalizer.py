import json
building_name = 'ebu3b'

with open("model/fe_%s.json"%building_name, 'r') as fp:
    data_feature_dict = json.load(fp)

feature_num = len(list(data_feature_dict.values())[0])
bin_total_num = 20
bin_size = int(len(data_feature_dict)/bin_total_num)

bin_range_list_list = list()
for i in range(0,feature_num):
    bin_range_list = list()
    sorted_features = sorted([feature[i] 
                            for feature 
                            in data_feature_dict.values()])
    bin_begin = sorted_features[0]
    bin_begin_idx = 0
    bin_cnt = 1
    for j, feat in enumerate(sorted_features):
        if j==0:
            continue
        elif j > bin_begin_idx+bin_size and sorted_features[j-1]!=feat:
            bin_range_list.append((bin_begin, feat))
            bin_begin = feat
            bin_begin_idx = j
            #for j in range(0,bin_num):
    bin_range_list.append((bin_begin, sorted_features[-1]+1))
    bin_range_list_list.append(bin_range_list)
            
def find_bin(bin_range_list, val):
    for i, bin_range in enumerate(bin_range_list):
        if val >= bin_range[0] and val<bin_range[1]:
            return float(i) / len(bin_range_list)
        
normalized_data_feature_dict = dict()
for srcid, features in data_feature_dict.items():
    normalized_data_feature_dict[srcid] = \
            [find_bin(bin_range_list, feat) for bin_range_list, feat \
            in zip(bin_range_list_list, features)]


with open('model/fe_%s_normalized.json'%building_name, 'w') as fp:
    json.dump(normalized_data_feature_dict, fp, indent=4)
