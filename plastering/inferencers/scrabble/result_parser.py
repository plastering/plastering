import json
import argparse
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)
    
parser = argparse.ArgumentParser()
parser.add_argument('-f',
                    type=str,
                    help='Result filename to summarize the result',
                    dest='filename')
parser.add_argument('-exp',
                    type=str,
                    help='Result filename to summarize the result',
                    dest='filename')
parser.add_argument(choices=['learn_crf', 'predict_crf', 'entity', 'crf_entity', \
                                 'init', 'result', 'iter_crf'],
                        dest = 'prog')

args = parser.parse_args()

filename = args.filename

with open(filename, 'r') as fp:
    data = json.load(fp)


if args.prog == 'crf_entity':
    source_nums = []
    char_f1s = []
    char_macrof1s = []
    entity_accuracies = []
    entity_macrof1s = []
    for datum in data:
        source_nums.append(dict())
        for building, learning_srcids in datum['source_list'].items():
            source_nums[-1][building] = len(learning_srcids)
        char_f1s.append(datum['result']['crf']['char_weighted_f1']) # TODO:validate
        char_macrof1s.append(datum['result']['crf']['char_macro_f1'])
        entity_accuracies.append(datum['result']['entity']['accuracy'])
        entity_macrof1s.append(datum['result']['entity']['macro_f1'])

    print('Source Nums')
    pp.pprint(source_nums)
    print('=================')
    print('Char F1s')
    print(char_f1s)
    print('=================')
    print('Char MacroF1s')
    print(char_macrof1s)
    print('=================')
    print('Entity Accuracies')
    print(entity_accuracies)
    print('=================')
    print('Entity MacroF1s')
    pp.pprint(entity_macrof1s)
    print('=================')

elif args.prog == 'iter_crf':
    source_nums = []
    char_f1s = []
    char_macrof1s = []
    for datum in data:
        source_nums.append(dict())
        for building, learning_srcids in datum['source_list'].items():
            source_nums[-1][building] = len(learning_srcids)
        char_f1s.append(datum['result']['crf']['char_weighted_f1']) # TODO:validate
        char_macrof1s.append(datum['result']['crf']['char_macro_f1'])

    print('Source Nums')
    pp.pprint(source_nums)
    print('=================')
    print('Char F1s')
    print(char_f1s)
    print('=================')
    print('Char MacroF1s')
    print(char_macrof1s)
    print('=================')
