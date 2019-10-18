
# coding: utf-8

# In[1]:

from functools import reduce                                                    
import pycrfsuite                                                               
import json                                                                     
import pandas as pd                                                             
import numpy as np                                                              
#import brick_parser                                                            
#reload(brick_parser)                                                           
#from brick_parser import tagList, tagsetList, equipTagsetList, pointTagsetList, locationTagsetList,equalDict, pointTagList, equipTagList, locationTagList, equipPointDict
import random


# In[2]:

buildingName = 'ebu3b'
with open('metadata/%s_label_dict.json'%buildingName, 'r') as fp:
    labelListDict = json.load(fp)
with open("metadata/%s_sentence_dict.json"%buildingName, "r") as fp:
    sentenceDict = json.load(fp)



# In[3]:

adder = lambda x,y:x+y
totalWordSet = set(reduce(adder, sentenceDict.values(), []))


# In[4]:

#def calc_features(sentence,labels):
def calc_features(sentence):
    sentenceFeatures = list()
    for i, word in enumerate(sentence):
        features = [
            'word.lower=' + word.lower(),
            'word.isdigit=%s' % word.isdigit()
        ]
        if i>0:
            features.extend([
                    '-1:word.lower=' + sentence[i-1].lower()
                ])
        else:
            features.append('BOS')
            
        if i<len(sentence)-1:
            features.extend([
                    '+1:word.lower=' + sentence[i+1].lower(),
                ])
        else:
            features.append('EOS')
        sentenceFeatures.append(features)
    return sentenceFeatures


# In[5]:

iter_num = 15
sample_num = 300
precision_list = list()
for c in range(0,iter_num):
    print(c)
    #%%time
    trainer = pycrfsuite.Trainer(verbose=False)
    #for srcid, setence in sentenceDict.items():

    randomIdxList = random.sample(range(0,len(labelListDict)),sample_num)
    for i, (srcid, labels) in enumerate(labelListDict.items()):
        if i not in randomIdxList:
            continue
        sentence = sentenceDict[srcid]
        #trainer.append(pycrfsuite.ItemSequence(calc_features(sentence, labels)), labels)
        trainer.append(pycrfsuite.ItemSequence(calc_features(sentence)), labels)


    # In[6]:

    #%%time
    trainer.train('random.crfsuite')


    # In[7]:

    tagger = pycrfsuite.Tagger()
    tagger.open('random.crfsuite')


    # In[8]:

    #%%time
    predictedDict = dict()
    for srcid, sentence in sentenceDict.items():
        predicted = tagger.tag(calc_features(sentence))
        predictedDict[srcid] = predicted


    # In[9]:

    DEBUG = False
    precisionOfTrainingDataset = 0.0                                                
    totalWordCount = 0.0                                                            
    labeledSrcidList = list(labelListDict.keys())
    randomSrcidList = [labeledSrcidList[idx] for idx in randomIdxList]

    for srcid in labelListDict.keys():                                              
        if srcid in randomSrcidList:
            continue
    #for i, srcid in enumerate(sentenceDict.keys()):                                
        #if not i in randIdxList:                                                   
        #    continue                                                               
        if DEBUG:
            print("===================== %s ========================== "%srcid)         
        sentence = sentenceDict[srcid]                                              
        predicted = predictedDict[srcid]                                            
        if not srcid in labelListDict.keys():                                       
            for word, predTag in zip(sentence, predicted):                          
                print('{:20s} {:20s}'.format(word,predTag))                         
        else:                                                                       
            printing_pairs = list()                                                 
            for word, predTag, origLabel \
                    in zip(sentence, predicted, labelListDict[srcid]):
                printing_pair = [word,predTag,origLabel]                            
                if not origLabel in ['none', 'identifier', 'network_adapter', 'rm']: 
                    if predTag==origLabel:                                          
                        precisionOfTrainingDataset += 1                             
                        printing_pair = ['O'] + printing_pair                       
                    else:                                                           
                        printing_pair = ['X'] + printing_pair                                       
                    totalWordCount += 1                                             
                else:                                                               
                    printing_pair = ['O'] + printing_pair                           
                printing_pairs.append(printing_pair)                                
            if 'X' in [pair[0] for pair in printing_pairs]:                         
                for (flag, word, predTag, origLabel) in printing_pairs:
                    if DEBUG:
                        print('{:5s} {:20s} {:20s} {:20s}'\
                                .format(flag, word, predTag, origLabel))                                          

    precision_list.append(precisionOfTrainingDataset/totalWordCount)


print("# Learning Sample: ",len(randomSrcidList))
print("# Learning Sample: ",len(labelListDict) - len(randomSrcidList))
print("Precision (mean): ", np.mean(precision_list))
print("Precision (std. dev.): ", np.std(precision_list))
