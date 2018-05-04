# coding: utf-8

# In[45]:

from functools import reduce
import pdb
import code

import rdflib
from rdflib.namespace import RDFS
from rdflib import URIRef, BNode, Literal
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

from termcolor import colored

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import OWL, RDF, RDFS
import rdflib
import arrow

from copy import deepcopy

pointPostfixes = ['alarm', 'sensor', 'setpoint', 'command', 'status', 'meter']
equipPostfixes = ['system', 'dhws', 'tower', 'chiller', 'coil', 'fan',
                       'hws', 'storage', 'battery', 'condenser', 'unit', 'fcu',
                       'vav', 'volume', 'economizer', 'hood', 'filter', 'vfd',
                       'valve', 'condensor', 'damper', 'hx', 'exchanger',
                       'thermostat', 'ahu', 'drive', 'heater', 'pump',
                       'conditioning', 'ws', 'dhws', 'elevator', 'fcp',
                       'panel', 'weather', 'generator', 'inverter', 'response',
                       'cws', 'crac', 'equipment', 'hvac']


# TODO: Check if a file is parsed or not and then load it or execute below.
parsed_files = ['brick/tags.json', \
#                'brick/tagsets.json', \
                'brick/equip_tagsets.json',\
                'brick/location_tagsets.json',\
                'brick/point_tagsets.json',\
                'brick/location_subclass_dict.json',\
                'brick/point_subclass_dict.json',\
                'brick/equip_subclass_dict.json',\
                'brick/tagset_tree.json',\
               ]
if False not in [os.path.isfile(fn) for fn in parsed_files]:
#if False:
    with open('brick/tags.json', 'r') as fp:
        tagList = json.load(fp)
#    with open('brick/tagsets.json', 'r') as fp:
#        tagsetList = json.load(fp)
    with open('brick/equip_tagsets.json', 'r') as fp:
        equipTagsetList = json.load(fp)
    with open('brick/location_tagsets.json', 'r') as fp:
        locationTagsetList = json.load(fp)
    with open('brick/point_tagsets.json', 'r') as fp:
        pointTagsetList = json.load(fp)
    with open('brick/equip_subclass_dict.json', 'r') as fp:
        equipSubclassDict = json.load(fp)
    with open('brick/location_subclass_dict.json', 'r') as fp:
        locationSubclassDict = json.load(fp)
    with open('brick/point_subclass_dict.json', 'r') as fp:
        pointSubclassDict = json.load(fp)
    with open('brick/tagset_tree.json', 'r') as fp:
        tagsetTree = json.load(fp)
else:
    def lcs_len(X, Y):
            m = len(X)
            n = len(Y)
            # An (m+1) times (n+1) matrix
            C = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m+1):
                    for j in range(1, n+1):
                            if X[i-1] == Y[j-1]: 
                                    C[i][j] = C[i-1][j-1] + 1
                            else:
                                    C[i][j] = max(C[i][j-1], C[i-1][j])
            lenList = [subC[-1] for subC in C]
            return max(lenList)


    #### Queries###
    ##############
    updateQueryPrefix= """
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX brick: <https://brickschema.org/schema/1.0.1/Brick#>
            INSERT DATA {
            """
    oneContent = """owl:equivalentClass {0} ;
    """
    subclassContent = """rdfs:subClassOf {0} .
    """

    updateQueryPostfix = """
    }"""

    base_query = lambda where_clause: """
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX brick: <https://brickschema.org/schema/1.0.1/Brick#>
            SELECT ?anything
            WHERE{
                %s
            }
            """ %(where_clause)

    strictSubclassesQueryWhere = lambda subclassName: \
            """
            {?anything rdfs:subClassOf+ %s .}
            UNION
            {?sameclass owl:equivalentClass %s .
             ?anything rdfs:subClassOf+ ?sameclass .}
            UNION
            {%s owl:equivalentClass ?sameclass .
             ?anything rdfs:subClassOf+ ?sameclass .}
            """ % (subclassName, subclassName, subclassName)

    strictSubclassesQuery = lambda subclassName: base_query(
                            strictSubclassesQueryWhere(subclassName))

    #strictSubclassesQuery = lambda subclassName: ("""
    #        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    #        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    #        PREFIX brick: <https://brickschema.org/schema/1.0.1/Brick#>
    #        SELECT ?anything
    #        WHERE{
    #            ?anything rdfs:subClassOf+ %s.
    #        }
    #        """ %(subclassName)
    #                )
    subclassesQuery = lambda subclassName: ("""
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX brick: <https://brickschema.org/schema/1.0.1/Brick#>
            SELECT ?anything
            WHERE{
                ?anything rdfs:subClassOf* %s.
            }
            """ %(subclassName)
                    )

    directSubclassesQueryWhere = lambda subclassName: """
        {?anything rdfs:subClassOf %s .}
        UNION
        {?sameclass owl:equivalentClass %s .
         ?anything rdfs:subClassOf ?sameclass .}
        UNION
        {%s owl:equivalentClass ?sameclass .
         ?anything rdfs:subClassOf ?sameclass .}
         """ %(subclassName, subclassName, subclassName)
    directSubclassesQuery = lambda subclassName: base_query(
                            directSubclassesQueryWhere(subclassName))
    #directSubclassesQuery = lambda subclassName: ("""
    #        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    #        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    #        PREFIX brick: <https://brickschema.org/schema/1.0.1/Brick#>
    #        SELECT ?anything
    #        WHERE{
    #            ?anything rdfs:subClassOf %s.
    #        }
    #        """ %(subclassName)
    #                )
    superclassesQuery = lambda subclassName: ("""
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX brick: <https://brickschema.org/schema/1.0.1/Brick#>
            SELECT ?anything
            WHERE{
                %s rdfs:subClassOf+ ?anything.
            }
            """ %(subclassName)
                    )
    directSuperclassesQuery = lambda subclassName: ("""
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX brick: <https://brickschema.org/schema/1.0.1/Brick#>
            SELECT ?anything
            WHERE{
                %s rdfs:subClassOf ?anything .
            }
            """ %(subclassName)
                    )



    # In[46]:

    BRICK      = Namespace('https://brickschema.org/schema/1.0.1/Brick#')
    BRICKFRAME = Namespace('https://brickschema.org/schema/1.0.1/BrickFrame#')
    BRICKTAG   = Namespace('https://brickschema.org/schema/1.0.1/BrickTag#')

    BUILDING = Namespace('https://brickschema.org/schema/1.0.1/building_example#')

    g = Graph()
    print("Init Graph")
    g.parse('/home/jbkoh/repo/Brick_jbkoh/dist/Brick.ttl', format='turtle')
    #g.parse('../GroundTruth/Brick/Brick.ttl', format='turtle')
    print("Load Brick.ttl")
    g.parse('../GroundTruth/Brick/BrickFrame.ttl', format='turtle')
    #g.parse('Brick/BrickTag.ttl', format='turtle')
    print("Load BrickFrame.ttl")

    g.bind('rdf'  , RDF)
    g.bind('rdfs' , RDFS)
    g.bind('brick', BRICK)
    g.bind('bf'   , BRICKFRAME)
    g.bind('btag' , BRICKTAG)


    def calc_leave_depth(tree, d=dict(), depth=0):
        curr_depth = depth + 1
        for tagset, branches in tree.items():
            d[tagset] = curr_depth
            for branch in branches:
                d.update(calc_leave_depth(branch, d, curr_depth))
        return d


    def printResults(res):
        if len(res) > 0:
            color = 'green'
        else:
            color = 'red'
        print(colored("-> {0} results".format(len(res)), color, attrs=['bold']))

    def printTuples(res):
        for row in res:
            #print map(lambda x: x.split('#')[-1], row)
            print(row[0])

    from collections import Counter

    def get_direct_superclasses(g, entity_name):
        res = g.query(directSuperclassesQuery(entity_name))
        super_tagsets = list()
        for row in res:
            thing = row[0]
            tagset = thing.split('#')[-1].lower()
            super_tagsets.append(tagset)
        return super_tagsets

    def extract_all_subclasses(g, subclassName, sparql_query=subclassesQuery, rawFlag=False):
            subclassList = list()
            res = g.query(sparql_query(subclassName))
            for row in res:
                    thing = row[0]
                    if rawFlag:
                            subclass = thing.split('#')[-1]
                    else:
                            subclass = thing.split('#')[-1].lower()
                    subclassList.append(subclass)
            try:
                    assert(len(subclassList)==len(set(subclassList)))
            except:
                    print('------------------------')
                    print(len(subclassList))
                    print(len(set(subclassList)))
                    subclassCounter = Counter(subclassList)
                    for subclass,cnt in subclassCounter.items():
                            if cnt>1:
                                    print(subclass)
                    print('------------------------')
                    assert(False)
            return subclassList

    def construct_subclass_tree(g, head, tagset_type):
        upper_tagset = head.split(':')[-1].lower()
        res = g.query(directSubclassesQuery(head))
        subclasses = list()
        tagsets = list()
        branches = list()
        for row in res:
            thing = row[0]
            subclass = thing.split('#')[-1]
            tagset = subclass.lower()
            if tagset_type == 'point' and tagset.split('_')[-1]\
               not in pointPostfixes:
                continue
            if tagset_type == 'equip' and tagset.split('_')[-1] \
               not in equipPostfixes:
                continue

            subclasses.append(subclass)
            tagsets.append(tagset)
            branches.append(construct_subclass_tree(g, 'brick:'+subclass, \
                                                    tagset_type))
        tree = {upper_tagset: branches}
        return tree

    def extract_all_subclass_tree(g, subclassName, tagsetType, rawFlag=False):
            subclassDict = dict()
#            res = g.query(strictSubclassesQuery(subclassName))
            subclassTagset = subclassName.split(':')[1].lower()
            tagsetList = extract_all_subclasses(g, subclassName, \
                                                sparql_query=strictSubclassesQuery,\
                                                rawFlag=True)
            tagsetList = [tagset for tagset in tagsetList if\
                          (tagsetType=='equip' and tagset.split('_')[-1].lower() in equipPostfixes) or
                          (tagsetType=='point' and tagset.split('_')[-1].lower() in pointPostfixes) or
                          (tagsetType=='location')]
            subclassDict[subclassTagset] = [tagset.lower() \
                                            for tagset in tagsetList]
            for tagset in tagsetList:
                subclassDict.update(\
                    extract_all_subclass_tree(g, 'brick:'+tagset, tagsetType))
            return subclassDict

    def extract_all_superclasses(g, subclassName, rawFlag=False):
        superclassList = list()
        res = g.query(superclassesQuery(subclassName))
        for row in res:
            thing = row[0]
            if rawFlag:
                subclass = thing.split('#')[-1]
            else:
                subclass = thing.split('#')[-1].lower()
                superclassList.append(subclass)
            try:
                assert(len(superclassList)==len(set(superclassList)))
            except:
                print('------------------------')
                print(len(superclassList))
                print(len(set(superclassList)))
                subclassCounter = Counter(superclassList)
                for subclass,cnt in subclassCounter.items():
                    if cnt>1:
                        print(subclass)
                    print('------------------------')
                    assert(False)
            return superclassList

    equalDict = dict()
    for s,p,o in g:
        if p==OWL.equivalentClass:
            a = s.split('#')[-1].lower()
            b = o.split('#')[-1].lower()
            equalDict[a] = b
            equalDict[b] = a



    pointTagsetList = extract_all_subclasses(g, "brick:Alarm")+ \
                                    extract_all_subclasses(g, "brick:Command")+\
                                    extract_all_subclasses(g, "brick:Meter")+\
                                    extract_all_subclasses(g, "brick:Sensor")+\
                                    extract_all_subclasses(g, "brick:Status")+\
                                    extract_all_subclasses(g, "brick:Timer")+\
                                    extract_all_subclasses(g, "brick:Setpoint")

    content = """
    """
    cnt = 0
    entity_maker = lambda tagset: '_'.join([tag[0].upper() + tag[1:] for tag in tagset.split('_')])
    for pointTagset in pointTagsetList:
            origPointEntityName = 'brick:' + entity_maker(pointTagset)
            newPointEntityName = 'brick:'
            if 'supply' in pointTagset:
                    newPointTagset = pointTagset.replace('supply', 'discharge')
                    if newPointTagset not in pointTagsetList:
                            pointTagsetList.append(newPointTagset)
                            newPointEntityName += entity_maker(newPointTagset)
            if 'discharge' in pointTagset:
                    newPointTagset = pointTagset.replace('discharge', 'supply')
                    if newPointTagset not in pointTagsetList:
                            pointTagsetList.append(newPointTagset)
                            newPointEntityName += entity_maker(newPointTagset)
            if newPointEntityName != 'brick:':
                cnt += 1
                res = g.query(directSuperclassesQuery(origPointEntityName))
                for row in res:
                    superclass = 'brick:' + row[0].split('#')[-1].split(':')[-1]
                    break
                content += """{0} a owl:Class ;
                """.format(newPointEntityName)
                content += oneContent.format(origPointEntityName)
                content += subclassContent.format(superclass)
    print(cnt)

    updateQuery = updateQueryPrefix + content + updateQueryPostfix
    g.update(updateQuery)
    
    equipTagsetList = extract_all_subclasses(g, "brick:Equipment")
    locationTagsetList = extract_all_subclasses(g, "brick:Location")
    measureTagsetList = extract_all_subclasses(g, "brick:MeasurementProperty")


    # TODO: If it is worse, remove this
    resourceTagsetList = extract_all_subclasses(g, "brick:Resource")

    removingTagsetList = list()
    usingAcronymList = ['hvac', 'vav', 'ahu', 'vfd', 'crac']
    for tagset in pointTagsetList + equipTagsetList + locationTagsetList + measureTagsetList + resourceTagsetList:
            if tagset in equalDict.keys():
                    pass
                    """ TODO: validate if these removing acronym helps.
                    if tagset in usingAcronymList:
                            removingTagsetList.append(equalDict[tagset])
                    else:
                            #if len(tagset)<len(equalDict[tagset]) and lcs_len(tagset, equalDict[tagset])==len(tagset):
                            if len(tagset)*2<len(equalDict[tagset]):
                                    removingTagsetList.append(tagset)
                    """
    for tagset in removingTagsetList:
            try:
                    pointTagsetList.remove(tagset)
            except:
                    pass
            try:
                    equipTagsetList.remove(tagset)
            except:
                    pass
            try:
                    locationTagsetList.remove(tagset)
            except:
                    pass
            try:
                    measureTagsetList.remove(tagset)
            except:
                    pass
            try:
                    ResourceTagsetList.remove(tagset)
            except:
                    pass

    pointTagsetList = [tagset for tagset in pointTagsetList if 'glycool' not in tagset]
    pointTagsetList = [tagset for tagset in pointTagsetList\
                       if tagset.split('_')[-1] in pointPostfixes]
    equipTagsetList = [tagset for tagset in equipTagsetList\
                       if tagset.split('_')[-1] in equipPostfixes]

    with open('brick/point_tagsets.json', 'w') as fp:
        json.dump(pointTagsetList, fp, indent=2)

    # validation code to find incorrect subclass relationship
    for tagset in pointTagsetList:
        entity = '_'.join([tag[0].upper() + tag[1:] \
                           for tag in tagset.split('_')])
        super_tagsets = get_direct_superclasses(g, 'brick:' + entity)
        if len(super_tagsets)>1:
            """
            print('\n')
            print(tagset)
            print(super_tagsets)
            pdb.set_trace()
            """
    adder = lambda x, y: x + y

    pointSubclassDict = dict()
    beginTime = arrow.get()
    for head in ['Sensor', 'Alarm', 'Status', 'Setpoint', 'Command', 'Meter']:
        pointSubclassDict.update(extract_all_subclass_tree(g, 'brick:'+head, 'point'))
    for tagset in set(reduce(adder, pointSubclassDict.values(), [])):
        if not pointSubclassDict.get(tagset):
            pointSubclassDict[tagset] = []
    endTime = arrow.get()
    with open('brick/point_subclass_dict.json', 'w') as fp:
        json.dump(pointSubclassDict, fp, indent=2)
    print('PointSubclassDict construction time: {0}'.format(endTime-beginTime))

    beginTime = arrow.get()
    equipSubclassDict = extract_all_subclass_tree(g, 'brick:Equipment', 'equip')
    for tagset in set(reduce(adder, equipSubclassDict.values(), [])):
        if not equipSubclassDict.get(tagset):
            equipSubclassDict[tagset] = []
    with open('brick/equip_subclass_dict.json', 'w') as fp:
        json.dump(equipSubclassDict, fp, indent=2)
    endTime = arrow.get()

    print('EquipmentSubclassDict construction time: {0}'.format(endTime-beginTime))
    beginTime = arrow.get()
    locationSubclassDict = extract_all_subclass_tree(g, 'brick:Location', 'location')
    endTime = arrow.get()
    for tagset in set(reduce(adder, locationSubclassDict.values(), [])):
        if not locationSubclassDict.get(tagset):
            locationSubclassDict[tagset] = []
    print('LocationSubclassDict construction time: {0}'.format(endTime-beginTime))
    with open('brick/location_subclass_dict.json', 'w') as fp:
        json.dump(locationSubclassDict, fp, indent=2)


    tagsetTree = dict()
    for head in ['Sensor', 'Alarm', 'Status', 'Setpoint', 'Command', 'Meter']:
        tagsetTree.update(construct_subclass_tree(g, 'brick:'+head, 'point'))
    for head in ['Equipment']:
        tagsetTree.update(construct_subclass_tree(g, 'brick:'+head, 'equip'))
    for head in ['Location']:
        tagsetTree.update(construct_subclass_tree(g, 'brick:'+head, 'location'))
    with open('brick/tagset_tree.json', 'w') as fp:
        json.dump(tagsetTree, fp, indent=2)

    depth_dict = calc_leave_depth(tagsetTree)
    with open('brick/tree_depth_dict.json', 'w') as fp:
        json.dump(depth_dict, fp, indent=2)

    beginTime = arrow.get()
    #TODO: Validate if Sensor is detected by this.

    tagsetList = pointTagsetList + equipTagsetList + locationTagsetList + measureTagsetList + resourceTagsetList
    separater = lambda s:s.split('_')
    tagList = list(set(reduce(lambda x,y: x+y, map(separater,tagsetList))))
    equipTagList = list(set(reduce(lambda x,y: x+y, map(separater,equipTagsetList))))
    pointTagList = list(set(reduce(lambda x,y: x+y, map(separater,pointTagsetList))))
    locationTagList = list(set(reduce(lambda x,y: x+y, map(separater,locationTagsetList))))
    measureTagList = list(set(reduce(lambda x,y: x+y, map(separater,measureTagsetList))))

    if '' in tagList:
       tagList.remove('')


    equipPointDict = dict()
    origEquipTagsetList = extract_all_subclasses(g, "brick:Equipment", rawFlag=True)
    for equipTagset in origEquipTagsetList:
            correspondingPointList = list()
            queryEquipTagset = ':'+equipTagset
            query = """
            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX brick: <https://brickschema.org/schema/1.0.1/Brick#>
                    PREFIX bf: <https://brickschema.org/schema/1.0.1/BrickFrame#>
            SELECT ?point
            WHERE{
                            BIND (%s AS ?start)
                            ?start rdfs:subClassOf ?par .
                            ?par a owl:Restriction .
                            ?par owl:onProperty bf:hasPoint .
                            ?par owl:someValuesFrom ?point .
                            }
            """ %queryEquipTagset
            res = g.query(query)
            for row in res:
                    thing = row[0]
                    point = thing.split('#')[-1].lower()
                    point = point.split(':')[-1]
                    correspondingPointList.append(point)
            if len(correspondingPointList)==0:
                    continue
            equipTagsetName = equipTagset.lower()
            if not equipTagsetName in equipPointDict.keys():
                    equipPointDict[equipTagsetName] = correspondingPointList
            if equipTagsetName in equalDict.keys():
                    if not  equalDict[equipTagsetName] in equipPointDict.keys():
                            equipPointDict[equalDict[equipTagsetName]] = correspondingPointList
    for equipTagset in equipTagsetList:
            if not equipTagset in equipPointDict.keys():
                    equipPointDict[equipTagset] = list()

    for equipTagsetName, correspondingPointList in equipPointDict.items():
            for pointTagset in correspondingPointList:
                    if 'supply' in pointTagset:
                            newPointTagset = pointTagset.replace('supply', 'discharge')
                            if newPointTagset not in correspondingPointList:
                                    correspondingPointList.append(newPointTagset)
                    if 'discharge' in pointTagset:
                            newPointTagset = pointTagset.replace('discharge', 'supply')
                            if newPointTagset not in correspondingPointList:
                                    correspondingPointList.append(newPointTagset)



    equalDict['co2_level_sensor'] = "co2_sensor"
    equalDict['co2_sensor'] = "co2_level_sensor"
    equalDict['room_temperature_setpoint'] = 'zone_temperature_setpoint'
    equalDict['temperature_setpoint'] = 'zone_temperature_setpoint'
    equalDict['zone_temperature_setpoint'] = 'temperature_setpoint'
    equalDict['zone_temperature_setpoint'] = 'room_temperature_setpoint'
    equalDict['temperature_setpoint'] = 'room_temperature_setpoint'
    equalDict['room_temperature_setpoint'] = 'temperature_setpoint'
    equalDict['effective_cooling_temperature_setpoint'] = 'cooling_temperature_setpoint'
    equalDict['cooling_temperature_setpoint'] = 'effective_cooling_temperature_setpoint'
    equalDict['effective_heating_temperature_setpoint'] = 'heating_temperature_setpoint'
    equalDict['heating_temperature_setpoint'] = 'effective_heating_temperature_setpoint'


    equipRelationDict = defaultdict(set)
    equipRelationDict['ahu'] = set(['chilled_water_system','hot_water_system', 'heat_exchanger', 'economizer', 'supply_fan', 'return_fan', 'exhaust_fan', 'mixed_air_filter',\
                                                    'mixed_air_damper', 'outside_air_damper', 'return_air_damper', 'cooling_coil', 'heating_coil', 'vfd'])
    equipRelationDict['vav'] = set(['vav', 'reheat_valve', 'damper', 'booster_fan', 'vfd'])
    equipRelationDict['supply_fan'] = set(['vfd', 'ahu'])
    equipRelationDict['return_fan'] = set(['vfd', 'ahu'])
    equipRelationDict['chilled_water_system'] = set(['chilled_water_pump', 'vfd', 'ahu'])
    equipRelationDict['hot_water_system'] = set(['hot_water_pump', 'vfd', 'ahu'])

    equipRelationDict['vfd'] = set(['supply_fan', 'return_fan', 'ahu', 'chilled_water_pump', 'hot_water_pump', 'chilled_water_system', 'hot_water_system', 'vav'])
    equipRelationDict['chilled_water_pump'] = set(['chilled_water_system', 'ahu', 'vfd'])
    equipRelationDict['hot_water_pump'] = set(['hot_water_system', 'ahu', 'vfd'])


    for equip, subEquipList in list(equipRelationDict.items()):
            for subEquip in subEquipList:
                    equipRelationDict[subEquip].add(equip)

    equipRelationDict = dict(equipRelationDict)
    for equip, subEquipList in equipRelationDict.items():
            equipRelationDict[equip] = list(subEquipList)

    locationRelationDict = dict()
    locationRelationDict['basement'] = ['room']
    locationRelationDict['floor'] = ['room']
    locationRelationDict['building'] = ['basement','floor','room']
    locationRelationDict['room'] = ['basement', 'floor', 'building']

print('Brick Loaded')

if __name__=='__main__':
    with open('brick/tags.json', 'w') as fp:
        json.dump(tagList, fp, indent=2)
    with open('brick/tagsets.json', 'w') as fp:
        json.dump(tagsetList, fp, indent=2)
    with open('brick/equip_tagsets.json', 'w') as fp:
        json.dump(equipTagsetList, fp, indent=2)
    with open('brick/location_tagsets.json', 'w') as fp:
        json.dump(locationTagsetList, fp, indent=2)
