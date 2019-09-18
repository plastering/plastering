import shelve
import numpy as np
import re
import pandas as pd
import json
import pickle
import pdb
from copy import copy

with open('metadata/bacnet_devices.json', 'r') as fp:
    sensor_dict = json.load(fp)

nae_dict = dict()
nae_dict['bonner'] = ["607", "608", "609", "557", "610"]
nae_dict['ap_m'] = ['514', '513','604']
nae_dict['bsb'] = ['519', '568', '567', '566', '564', '565']
nae_dict['ebu3b'] = ["505", "506"]
nae_dict['music'] = ['523']
nae_dict['sme'] = ['572', '573', '574']
nae_dict['bml'] = ['510']

# Vectorization

#Parse the data in the dictionary as filtered by device_list
#Gives us a sensor_list with sensor information of a building
def remove_emptystr(s):
        while '' in s:
                s.remove('')
        return s

def extract_words(sentence, delimiter):
#result = re.findall('(\d+\s?-?\s?\d+)|(\d+)', sentence)
        result = sentence.lower().split(delimiter)
        while '' in result:
                result.remove('')
        return result

def sentence2lower(wordList):
        return [word.lower() for word in wordList]

def tokenize(tokenType, raw, mappedWordMap=None):
        raw = raw.replace('_', ' ')
        if tokenType=='Alphanumeric':
                sentence = re.findall("\w+", raw)
        elif tokenType in ['AlphaAndNum', 'NumAsSingleWord']:
                sentence = re.findall("[a-zA-Z]+|\d+", raw)
        elif tokenType=='NoNumber':
                sentence = re.findall("[a-zA-Z]+", raw)
        elif tokenType=='JustSeparate':
            sentence = re.findall("([a-zA-Z]+|\d+|[^0-9a-z])", raw)
        else:
                assert(False)
        if tokenType=='NumAsSingleWord':
                sentence = ['NUM' if len(re.findall('\d+',word))>0 else word for word in sentence]
        sentence = sentence2lower(sentence)

        if mappedWordMap!=None:
                terms = mappedWordMap.keys()
        else:
                terms = list()

        retSentence = list()
        for word in sentence:
                alphaWord = re.findall('[a-zA-Z]+',word)
                if len(alphaWord )>0:
                        if alphaWord in terms:
                                #mappedWordList = mappedWordMap[alphaWord]
                                #for mappedWord in mappedWordList:
                                        #word = word.replace(mappedWord, '_'+mappedWord+'_')
                                word = word.replace(alphaWord, '_'+'_'.join(mappedWordMap[alphaWord])+'_')
                retSentence = retSentence + remove_emptystr(word.split('_'))
        return retSentence

def parse_sentence(sentence):
    return re.findall("([a-zA-Z]+|\d+|[^0-9a-z])", sentence.lower())

def get_bacnettype_dict(building_name):
    bacnettypeMap = pd.read_csv('metadata/bacnettype_mapping.csv').set_index('bacnet_type_str')
    naeList = nae_dict[building_name]

    source_id_set = set([])
    bacnettype_dict = dict()
    bacnettype_code_dict = dict()


    for nae in naeList:
        device = sensor_dict[nae]
        h_dev = device['props']
        for sensor in device['objs']:
            h_obj = sensor['props']
            source_id = str(h_dev['device_id']) + '_' + str(h_obj['type']) + '_' + str(h_obj['instance'])
#                        if not source_id in validSrcidList:
#                                continue
            if h_obj['type'] not in (0,1,2,3,4,5,13,14,19):
                    continue
            if source_id in source_id_set:
                    continue
            else:
                    source_id_set.add(source_id)

            if sensor['props']['type_str']:
                typeStr = bacnettypeMap.loc[sensor['props']['type_str']].tolist()[0]
                if type(typeStr)!=str:
                    if np.isnan(typeStr):
                        typeStr = ''
                    else:
                        print("Error in bacnettype map file")
                        assert(False)
            else:
                    typeStr = ''
            bacnettype_dict[source_id] = typeStr
            bacnettype_code_dict[source_id] = sensor['props']['type_str']

    return bacnettype_dict

def get_unit_dict(building_name):
    unitMap = pd.read_csv('metadata/unit_mapping.csv').set_index('unit')
    naeList = nae_dict[building_name]

    unit_code_dict = dict()
    unit_dict = dict()
    source_id_set = set([])

    for nae in naeList:
        device = sensor_dict[nae]
        h_dev = device['props']
        for sensor in device['objs']:
            h_obj = sensor['props']
            source_id = str(h_dev['device_id']) + '_' + str(h_obj['type']) + '_' + str(h_obj['instance'])
            #                        if not source_id in validSrcidList:
            #                                continue
            if h_obj['type'] not in (0,1,2,3,4,5,13,14,19):
                continue
            if source_id in source_id_set:
                continue
            else:
                source_id_set.add(source_id)

            if sensor['unit']:
                try:
                    unit_str = unitMap.loc[sensor['unit']].tolist()[0]
                    if type(unit_str) != str:
                        if np.isnan(unit_str):
                            unit_str = ''
                        else:
                            print("Error in unit map file")
                            assert(False)
                except:
                    print("===================")
                    print(sensor['unit'])
                    print(sensor)
                    print("===================")
                    assert(False)
            else:
                unit_str = ''
            unit_code_dict[source_id] = sensor['unit']
            unit_dict[source_id] = unit_str

    return unit_dict


def parse_sentences(building_name):
    if building_name == 'pouya':
        metadata_file = 'metadata/pouya.csv'
        df = pd.read_csv(metadata_file)
        names = df['Address'].tolist()
        srcids = copy(names)
        names = [parse_sentence(name) for name in names]
        blanks = ['' for name in names]
        jcinames = copy(blanks)
        descs = copy(blanks)
        units = copy(blanks)
        bacnettypes = copy(blanks)
        return None, srcids, names, jcinames, descs, units, bacnettypes 
    unitMap = pd.read_csv('metadata/unit_mapping.csv').set_index('unit')
    bacnettypeMap = pd.read_csv('metadata/bacnettype_mapping.csv').set_index('bacnet_type_str')
    naeList = nae_dict[building_name]

    sensor_list = []
    name_list = []
    desc_list = []
    unit_list = []
    bacnettype_list = []
    jciname_list = list()
    source_id_set = set([])
    source_id_list = list()

    for nae in naeList:
        device = sensor_dict[nae]
        h_dev = device['props']
        for sensor in device['objs']:
            h_obj = sensor['props']
            source_id = str(h_dev['device_id']) + '_' + str(h_obj['type']) + '_' + str(h_obj['instance'])
            #                        if not source_id in validSrcidList:
            #                                continue
            if h_obj['type'] not in (0,1,2,3,4,5,13,14,19):
                continue
            if source_id in source_id_set:
                continue
            else:
                source_id_set.add(source_id)
            source_id_list.append(source_id) 
            jciname_list.append(parse_sentence(sensor['jci_name']))
            name_list.append(parse_sentence(sensor['name']))
            desc_list.append(parse_sentence(sensor['desc']))
            if not sensor['unit']==None:
                try:
                    unit_str = unitMap.loc[sensor['unit']].tolist()[0]
                    if type(unit_str) != str:
                        if np.isnan(unit_str):
                            unit_str = ''
                        else:
                            print("Error in unit map file")
                            assert(False)
                except:
                    print("===================")
                    print(sensor['unit'])
                    print(sensor)
                    print("===================")
                    assert(False)
            else:
                unit_str = ''

            unit_list.append([unit_str])
            if not sensor['props']['type_str']==None:
                typeStr = bacnettypeMap.loc[sensor['props']['type_str']].tolist()[0]
                if type(typeStr)!=str:
                    if np.isnan(typeStr):
                        typeStr = ''
                    else:
                        print("Error in bacnettype map file")
                        assert(False)
            else:
                typeStr = ''
            bacnettype_list.append([typeStr])
            
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
    return sensor_df, source_id_list, name_list, jciname_list, desc_list, \
           unit_list, bacnettype_list



def structure_metadata(buildingName=None, tokenType=None, bigramFlag=False, validSrcidList=[], mappedWordMap=None, withDotFlag=True):

        unitMap = pd.read_csv('metadata/unit_mapping.csv').set_index('unit')
        bacnettypeMap = pd.read_csv('metadata/bacnettype_mapping.csv').set_index('bacnet_type_str')
        if bigramFlag:
                with open('data/bigrammer_'+buildingName+'_'+tokenType+'.pkl', 'rb') as fp:
                        bigrammer = pickle.load(fp)
        naeList = nae_dict[buildingName]

        sensor_list = []
        nameList = list()
        names_num_list = []
        names_str_list = []
        names_num_listWithDigits = [] 
        sensor_type_namez=[]
        descList = []
        unitList = []
        bacnettypeList = []
        type_str_list = []
        type_list = []
        jcinameList = list()
        jci_names_str_list = []
        source_id_set = set([])
        wordList = list()


        for nae in naeList:
                device = sensor_dict[nae]
                h_dev = device['props']
                for sensor in device['objs']:
                        h_obj = sensor['props']
                        source_id = str(h_dev['device_id']) + '_' + str(h_obj['type']) + '_' + str(h_obj['instance'])
#                        if not source_id in validSrcidList:
#                                continue
                        if h_obj['type'] not in (0,1,2,3,4,5,13,14,19):
                                continue
                        if source_id in source_id_set:
                                continue
                        else:
                                source_id_set.add(source_id)

                        jciSentence = tokenize(tokenType, sensor['jci_name'], mappedWordMap=mappedWordMap)
                        nameSentence = tokenize(tokenType, sensor['name'], mappedWordMap=mappedWordMap)
                        descSentence = tokenize(tokenType, sensor['desc'], mappedWordMap=mappedWordMap)

                        if not sensor['unit']==None:
                                try:
                                    unitStr = unitMap.loc[sensor['unit']].tolist()[0]
                                    if type(unitStr) != str:
                                            if np.isnan(unitStr):
                                                    unitStr = ''
                                            else:
                                                    print("Error in unit map file")
                                                    assert(False)
                                except:
                                    print("===================")
                                    print(sensor['unit'])
                                    print(sensor)
                                    print("===================")
                                    assert(False)
                        else:
                                unitStr = ''
                        unitSentence = tokenize(tokenType, unitStr)

                        if not sensor['props']['type_str']==None:
                                typeStr = bacnettypeMap.loc[sensor['props']['type_str']].tolist()[0]
                                if type(typeStr)!=str:
                                        if np.isnan(typeStr):
                                                typeStr = ''
                                        else:
                                                print("Error in bacnettype map file")
                                                assert(False)
                        else:
                                typeStr = ''
                        bacnettypeList.append(typeStr)


                        if bigramFlag:
                                jciSentence = bigrammer[jciSentence]
                                nameSentence = bigrammer[nameSentence]
                                descSentence = bigrammer[descSentence]

                        jcinameList.append(' '.join(jciSentence))
                        nameList.append(' '.join(nameSentence))
                        descList.append(' '.join(descSentence))
                        unitList.append(' '.join(unitSentence))

                        if withDotFlag:
                                wordList = wordList + jciSentence + ['DOT'] + nameSentence + ['DOT'] + descSentence +\
                                                   ['DOT'] + unitSentence + ['DOT', '\n']
                        else:
                                wordList = wordList + jciSentence + nameSentence + descSentence + unitSentence

#word = '5'#.upper()
#               infoTypeList = ['jci_name', 'name', 'desc']
#               for infoType in infoTypeList:
#                       if word in [token.lower() for token in re.findall("\w+", sensor[infoType])]:
#                       print sensor[infoType]


#convert string to dictionary for categorical vectorization
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
        sensor_df.set_index('source_id', inplace=True)
#sensor_df = sensor_df.set_index('source_id')
#sensor_df = sensor_df.groupby(sensor_df.index).first()
        return sensor_df, nameList, jcinameList, descList, unitList, bacnettypeList, wordList
