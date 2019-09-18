from jasonhelper import argparser


argparser.add_argument('-bl', type='slist', dest='buildings')
argparser.add_argument('-nl', type='ilist', dest='sample_nums')
argparser.add_argument('-t', type=str, dest='target_building')
argparser.add_argument('-step', type=int, dest='step_num')
argparser.add_argument('-iter', type=int, dest='iter_num')
