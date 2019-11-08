import os
import pdb
from uuid import uuid4
from operator import itemgetter
from pathlib import Path
from collections import defaultdict
import pprint
from rdflib import Graph, Namespace, RDF, URIRef
pp = pprint.PrettyPrinter(indent=4)

import pycrfsuite
from bson.binary import Binary as BsonBinary
import arrow
import numpy as np
import pandas as pd

from .mongo_models import store_model, get_model, get_tags_mapping, \
    get_crf_results, store_result, get_entity_results
from .base_scrabble import BaseScrabble
from .common import *

BRICK_VERSION = '1.0.1'
BASE = Namespace('http://example.com#')
BRICK = Namespace('https://brickschema.org/schema/{0}/Brick#'.format(BRICK_VERSION))

curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def check_tag_in_tagset(tag, tagset):
    if not tag:
        raise Exception('no tag detected')
    if not tagset:
        raise Exception('no tagset defined')
    tagset = tagset.split('_')
    tags = tag.split('_')
    in_flag = False
    ins = [t in tagset for t in tags]
    if False in ins:
        return False
    else:
        return True


def gen_uuid():
    return str(uuid4())


class Tagsets2Entities(BaseScrabble):
    def __init__(self,
                 target_building,
                 target_srcids,
                 building_label_dict,
                 building_sentence_dict,
                 building_tagsets_dict={},
                 source_buildings=[],
                 source_sample_num_list=[],
                 learning_srcids=[],
                 pgid=None,
                 config={}
                 ):
        super(Tagsets2Entities, self).__init__(
            target_building,
            target_srcids,
            building_label_dict,
            building_sentence_dict,
            {},
            source_buildings,
            source_sample_num_list,
            learning_srcids,
            config)
        self.model_uuid = None

        if 'crftype' in config:
            self.crftype = config['crftype']
        else:
            self.crftype = 'crfsuite'
        if 'query_strategy' in config:
            self.query_strategy = config['query_strategy']
        else:
            self.query_strategy = 'configidence'
        if 'user_cluster_flag' in config:
            self.use_cluster_flag = config['use_cluster_flag']
        else:
            self.use_cluster_flag = True

        self.building_tagsets_dict = building_tagsets_dict

        # Note: Hardcode to disable use_brick_flag
        """
        if 'use_brick_flag' in config:
            self.use_brick_flag = config['use_brick_flag']
        else:
            self.use_brick_flag = False  # Temporarily disable it
        """
        self.use_brick_flag = False
        self._init_data()

    def _init_data(self):
        self.sentence_dict = {}
        self.label_dict = {}
        self.tagsets_dict = {}
        for building, source_sample_num in zip(self.source_buildings, self.source_sample_num_list):
            self.sentence_dict.update(self.building_sentence_dict[building.id])
            one_label_dict = self.building_label_dict[building.id]
            self.label_dict.update(one_label_dict)
            one_tagsets_dict = self.building_tagsets_dict[building.id]
            self.tagsets_dict.update(one_tagsets_dict)

            if not self.learning_srcids:
                sample_srcid_list = select_random_samples(
                    building,
                    one_label_dict.keys(),
                    source_sample_num,
                    self.use_cluster_flag,
                    sentence_dict=self.building_sentence_dict[building.id]
                )
                self.learning_srcids += sample_srcid_list

        # Construct Brick examples
        brick_sentence_dict = dict()
        brick_label_dict = dict()
        if self.use_brick_flag:
            with open(curr_dir / 'metadata/brick_tags_labels.json', 'r') as fp:
                tag_label_list = json.load(fp)
            for tag_labels in tag_label_list:
                # Append meaningless characters before and after the tag
                # to make it separate from dependencies.
                # But comment them out to check if it works.
                # char_tags = [' '] + list(map(itemgetter(0), tag_labels)) + [' ']
                char_tags = list(map(itemgetter(0), tag_labels))
                # char_labels = ['O'] + list(map(itemgetter(1), tag_labels)) + ['O']
                char_labels = list(map(itemgetter(1), tag_labels))
                brick_sentence_dict[''.join(char_tags)] = char_tags + ['NEWLINE']
                brick_label_dict[''.join(char_tags)] = char_labels + ['O']
            self.sentence_dict.update(brick_sentence_dict)
            self.label_dict.update(brick_label_dict)
        self.brick_srcids = list(brick_sentence_dict.keys())

    def map_tags_tagsets(self):
        entities_dict = {}
        for srcid in self.target_srcids:
            tagsets = self.tagsets_dict[srcid]
            tags_dict = self.label_dict[srcid]
            entities = {}
            for tagset in tagsets:
                matched_words = []
                for metadata_type, tags in tags_dict.items():
                    sentence = self.sentence_dict[srcid][metadata_type]
                    tags = [tag[2:] if len(tag)>1 else tag for tag in tags]
                    used_indices = []
                    found_indices = [i for i, tag in enumerate(tags)
                                     if check_tag_in_tagset(tag, tagset)]
                    for i, tag in enumerate(tags):
                        if tag == 'leftidentifier':
                            """
                            prev_tag = 'O'
                            prev_idx = i
                            while prev_tag == 'O':
                                prev_idx += -1
                                prev_tag = tags[prev_idx]
                            if prev_idx in found_indices:
                                found_indices.append(i)
                            """
                            prev_tag = None
                            for prev_idx in reversed(range(0, i)):
                                prev_tag = tags[prev_idx]
                                if prev_tag != 'O': #TODO: This shouldn't be leftidentifier or rightidentifier either
                                    break
                            if prev_tag and prev_idx in found_indices:
                                found_indices.append(i)
                    for i, tag in reversed(list(enumerate(tags))):
                        if tag == 'rightidentifier':
                            next_tag = None
                            for next_idx in range(i+1, len(tags)):
                                next_tag = tags[next_idx]
                                if next_tag != 'O':
                                    break
                            if next_tag and next_idx in found_indices:
                                found_indices.append(i)
                    found_indices = sorted(found_indices)
                    matched_word = ''.join([sentence[i] for i
                                            in sorted(found_indices)])
                    if matched_word:
                        matched_words += [matched_word]
                    used_indices += found_indices
                entities[tagset] = '_'.join(matched_words)
            entities_dict[srcid] = entities
            pp.pprint(entities)
        with open('test.json', 'w') as fp:
            json.dump(entities_dict, fp, indent=2)
        return entities_dict

    def _init_graph(self):
        return Graph()

    def _make_instance_tuple(self, srcid, pred_point):
        return (URIRef(BASE + srcid), RDF.type, BRICK[pred_point])

    def _add_pred_point_result(self, pred_g, pred_confidences, srcid,
                               pred_point, pred_prob):
        triple = self._make_instance_tuple(srcid, pred_point)
        pred_confidences[triple] = pred_prob
        pred_g.add(triple)
        return pred_g, pred_confidences

    def graphize(self, entities_dict):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        rel_file = dir_path + '/../metadata/relationship_prior.json'
        with open(rel_file, 'r') as fp:
            rel_prior = json.load(fp)
        pred_g = self._init_graph()
        pred_confidences = {}
        for srcid, entities in entities_dict.items():
            tagsets = list(entities.keys())
            point_tagsets = list(find_points(tagsets))
            for point_tagset in point_tagsets:
                self._add_pred_point_result(pred_g, pred_confidences,
                                            srcid, point_tagset, 1)
            for tagset in tagsets:
                if tagset in point_tagsets:
                    continue
                # TODO: Below
                pass

        pred_g.serialize('test.ttl', format='turtle')
        return pred_g

if __name__ == '__main__':
    t2e = Tagsets2Entities()
