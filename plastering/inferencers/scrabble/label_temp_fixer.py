import json

with open('metadata/ebu3b_label_dict.json', 'r') as fp:
    label_dict = json.load(fp)
with open('metadata/ebu3b_sentence_dict.json', 'r') as fp:
    sentence_dict = json.load(fp)

new_label_dict = dict()
for srcid, labels in label_dict.items():
    sentence = sentence_dict[srcid]
    new_labels = list()
    for i, (word, label) in enumerate(zip(sentence, labels)):
        if word=='nae'or word=='n':
            new_labels.append('network_adapter')
        else:
            new_labels.append(label)
    new_label_dict[srcid] = new_labels

with open('metadata/ebu3b_label_dict2.json', 'w') as fp:
    label_dict = json.dump(new_label_dict, fp)
