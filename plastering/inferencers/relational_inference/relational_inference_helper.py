import argparse
from plastering.inferencers.relational_inference.util import read_config


# configuration setup
def parse_args_colocation():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-config', default='colocation', type=str)
    parser.add_argument('-model', default='stn', type=str,
                        choices=['stn'])
    parser.add_argument('-loss', default='comb', type=str,
                        choices=['triplet', 'comb'])
    parser.add_argument('-seed', default=2, type=int,
                        help="Random seed")
    parser.add_argument('-log', default='stn', type=str,
                        help="Log directory")
    parser.add_argument('-facility', default='Soda', type=int,
                        help="Log directory")
    parser.add_argument('-split', default='room', type=str,
                        help="split 1/5 sensors or rooms for test",
                        choices=['room', 'sensor'])
    args = parser.parse_args()
    # the file to be opened depends on where this method is called
    config = read_config('figs/' + args.config + '.yaml')
    return args, config


def parse_args_coequipment():
    parser = argparse.ArgumentParser(description='train_and_test.py')
    parser.add_argument('-config', default='coequipment', type=str)
    parser.add_argument('-task', default='coequipment', type=str,
                        choices=['colocation', 'coequipment'])
    parser.add_argument('-model', default='han', type=str,
                        choices=['han', 'basic'])
    parser.add_argument('-loss', default='triplet', type=str,
                        choices=['triplet', 'angular', 'softmax'])
    parser.add_argument('-seed', default=2, type=int,
                        help="Random seed")
    parser.add_argument('-log', default='coe', type=str,
                        help="Log directory")
    parser.add_argument('-facility', default='Soda', type=int,
                        help="Log directory")
    parser.add_argument('-split', default='room', type=str,
                        help="split 1/5 sensors or rooms for test",
                        choices=['room', 'sensor'])
    args = parser.parse_args()
    config = read_config('figs/' + args.config + '.yaml')
    return args, config
