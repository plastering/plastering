import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 8
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 8
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes

import argparse
import json
import subprocess
import pdb
from functools import reduce
import re
import numpy as np
import os
from operator import itemgetter
from collections import OrderedDict

import plotter
from plotter import save_fig


anon_building_dict = {
    'ebu3b': 'A-1',
    'bml': 'A-2',
    'ap_m': 'A-3',
    'ghc': 'B-1'
}

def interpolate(x1, y1, x2, y2):
    return (x1 * y2 + x2 * y1) / (x1 + x2)

def extrapolate(xt, x1, y1, x2, y2):
    return (xt*y2 - xt*y1 + x2*y1 - x1*y2) / (x2 - x1)


def lin_interpolated_avg(target_x, x_list, y_list):
    target_y = []
    for t_x in target_x:
        t_y_cands = []
        for given_x, given_y in zip(x_list, y_list):
            assert len(given_x) == len(given_y)
            left_x = -10000
            right_x = 10000
            left_y = None
            right_y = None
            left_y_idx = None
            right_y_dix = None
            for i, (one_x, one_y) in enumerate(zip(given_x, given_y)):
                if one_x <= t_x:
                    if one_x > left_x:
                        left_x = one_x
                        left_y = one_y
                        left_y_idx = i
                if one_x >= t_x:
                    if one_x < right_x:
                        right_x = one_x
                        right_y = one_y
                        right_y_idx = i
            if left_x == right_x and right_x == t_x:
                assert left_y == right_y
                t_y_cand = left_y
            else:
                if not left_y:
                    rr_x = given_x[right_y_idx + 1]
                    rr_y = given_y[right_y_idx + 1]
                    t_y_cand = extrapolate(t_x, right_x, right_y, rr_x, rr_y)
                elif not right_y:
                    ll_x = given_x[left_y_idx - 1]
                    ll_y = given_y[left_y_idx - 1]
                    t_y_cand = extrapolate(t_x, ll_x, ll_y, left_x, left_y)
                elif left_y and right_y:
                    t_y_cand = interpolate(t_x - left_x, left_y, 
                                           right_x - t_x, right_y)
                else:
                    assert False

            t_y_cands.append(t_y_cand)
        t_y = np.mean(t_y_cands)
        target_y.append(t_y)
    return target_y

def crf_result_acc():
    #source_target_list = [('ebu3b', 'ap_m'), ('ebu3b', 'ap_m')]
    source_target_list = [('ebu3b', 'ap_m'), ('ghc', 'ebu3b')]
    #n_list_list = [#[(1000, 0), (1000,5), (1000,20), (1000,50), (1000,100), (1000, 150), (1000,200)],
    #               [(200, 0), (200,5), (200,20), (200,50), (200,100), (200, 150), (200,200)],
    #               [(0,5), (0,20), (0,50), (0,100), (0,150), (0,200)]]
    char_precs_list = list()
    phrase_f1s_list = list()
#fig, ax = plt.subplots(1, 1)
    fig, axes = plt.subplots(1,len(source_target_list))
    if isinstance(axes, Axes):
        axes = [axes]
    fig.set_size_inches(4, 1.5)
    cs = ['firebrick', 'deepskyblue']
    filename_template = 'result/crf_iter_{0}_char2ir_iter_{1}.json'
    n_s_list = [1000, 200, 0]

    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = ['--', '-.', '-']
        plot_list = list()
        legends_list = list()
        for n_s in n_s_list:
            if n_s == 0:
                buildingfix = ''.join([target, target])
            else:
                buildingfix = ''.join([source, target, target])
            n = n_s + 0
            xs = [5] + list(range(10, 201, 10))
            x_cands = []
            f1_cands = []
            mf1_cands = []
            for exp_num in range(0,5):
                nfix = n + exp_num
                filename = filename_template.format(buildingfix, nfix)
                if not os.path.exists(filename):
                    pdb.set_trace()
                    continue
                with open(filename, 'r') as fp:
                    data = json.load(fp)
                x_cand = [len(datum['learning_srcids']) - n_s for datum in data]
                f1_cand = []
                for datum in data:
                    prec = datum['result']['crf']['phrase_precision'] * 100
                    rec = datum['result']['crf']['phrase_recall'] * 100
                    f1 = 2 * prec * rec / (prec + rec)
                    f1_cand.append(f1)
                mf1_cand = [datum['result']['crf']['phrase_macro_f1'] * 100
                            for datum in data]
                x_cands.append(x_cand)
                f1_cands.append(f1_cand)
                mf1_cands.append(mf1_cand)
            f1s = lin_interpolated_avg(xs, x_cands, f1_cands)
            mf1s = lin_interpolated_avg(xs, x_cands, mf1_cands)
            ys = [f1s]#, mf1s]
            # Print curr result
            if n_s == 200 or n_s == 0:
                print('=======')
                print(source, target, n_s)
                print('init F1: {0}'.format(f1s[0]))
                print('init MF1: {0}'.format(mf1s[0]))
                print('=======')

            xlabel = None
            ylabel = 'Score (%)'
            xtick = [5] + list(range(40, 205, 40))
            xtick_labels = [str(n) for n in xtick]
            ytick = range(0,101,20)
            ytick_labels = [str(n) for n in ytick]
            xlim = (-5, xtick[-1]+5)
            ylim = (ytick[0]-2, ytick[-1]+5)
            if i == 0:
                legends = [
                    '#$B_S$:{0}'.format(n_s),
                    #'#$B_S$:{0}'.format(n_s),
                ]
            else:
                legends = None

            title = None
            _, plots = plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                             xtick_labels, ytick, ytick_labels, title, ax, fig, \
                             ylim, xlim, legends, xtickRotate=0, \
                             linestyles=[linestyles.pop()]*len(ys), cs=cs)
            text = '{0} $\\Rightarrow$ {1}'.format(\
                    anon_building_dict[source],
                    anon_building_dict[target])
            ax.text(0.8, 0.1, text, transform=ax.transAxes, ha='right',
                    backgroundcolor='white'
                    )#, alpha=0)
            plot_list += plots

    axes[0].legend(bbox_to_anchor=(0.15, 0.96), ncol=3, frameon=False)
    for ax in axes:
        ax.grid(True)
    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    plt.text(0, 1.16, '$F_1$: \nMacro $F_1$: ', va='center', ha='center',
            transform=axes[0].transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center')

    save_fig(fig, 'figs/crf_acc.pdf')
    subprocess.call('./send_figures')


def crf_entity_result():
    building_sets = [('ebu3b', 'ap_m'), ('ap_m', 'bml'),
                 ('ebu3b', 'ghc'), ('ghc', 'ebu3b'), ('ebu3b', 'bml', 'ap_m')] ### TODO TODO: this should be changed to use ebu3b,ap_m -> bml
    #building_sets = [('ebu3b', 'ghc'), ('ebu3b', 'ghc')]
    #building_sets = [('ap_m',), ('bml',),
    #             ('ghc',), ('ebu3b',), ('ap_m',)] ### TODO TODO: this should be changed to use ebu3b,ap_m -> bml

def crf_result():
    #source_target_list = [('ebu3b', 'ap_m'), ('ebu3b', 'ap_m')]
    source_target_list = [('ebu3b', 'ap_m'), ('ghc', 'ebu3b')]
    #n_list_list = [#[(1000, 0), (1000,5), (1000,20), (1000,50), (1000,100), (1000, 150), (1000,200)],
    #               [(200, 0), (200,5), (200,20), (200,50), (200,100), (200, 150), (200,200)],
    #               [(0,5), (0,20), (0,50), (0,100), (0,150), (0,200)]]
    char_precs_list = list()
    phrase_f1s_list = list()
#fig, ax = plt.subplots(1, 1)
    fig, axes = plt.subplots(1,len(source_target_list))
    if isinstance(axes, Axes):
        axes = [axes]
    fig.set_size_inches(4, 1.5)
    cs = ['firebrick', 'deepskyblue']
    filename_template = 'result/crf_iter_{0}_char2ir_iter_{1}.json'
    n_s_list = [1000, 200, 0]

    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):
        linestyles = ['--', '-.', '-']
        plot_list = list()
        legends_list = list()
        for n_s in n_s_list:
            if n_s == 0:
                buildingfix = ''.join([target, target])
            else:
                buildingfix = ''.join([source, target, target])
            n = n_s + 0
            xs = [5] + list(range(10, 201, 10))
            x_cands = []
            f1_cands = []
            mf1_cands = []
            for exp_num in range(0,5):
                nfix = n + exp_num
                filename = filename_template.format(buildingfix, nfix)
                if not os.path.exists(filename):
                    continue
                with open(filename, 'r') as fp:
                    data = json.load(fp)
                x_cand = [len(datum['learning_srcids']) - n_s for datum in data]
                f1_cand = []
                for datum in data:
                    prec = datum['result']['crf']['phrase_precision'] * 100
                    rec = datum['result']['crf']['phrase_recall'] * 100
                    f1 = 2 * prec * rec / (prec + rec)
                    f1_cand.append(f1)
                mf1_cand = [datum['result']['crf']['phrase_macro_f1'] * 100 
                            for datum in data]
                x_cands.append(x_cand)
                f1_cands.append(f1_cand)
                mf1_cands.append(mf1_cand)
            f1s = lin_interpolated_avg(xs, x_cands, f1_cands)
            mf1s = lin_interpolated_avg(xs, x_cands, mf1_cands)
            ys = [f1s, mf1s]
            # Print curr result
            if n_s == 200 or n_s == 0:
                print('=======')
                print(source, target, n_s)
                print('init F1: {0}'.format(f1s[0]))
                print('init MF1: {0}'.format(mf1s[0]))
                print('=======')

            xlabel = None
            ylabel = 'Score (%)'
            xtick = [5] + list(range(40, 205, 40))
            xtick_labels = [str(n) for n in xtick]
            ytick = range(0,101,20)
            ytick_labels = [str(n) for n in ytick]
            xlim = (-5, xtick[-1]+5)
            ylim = (ytick[0]-2, ytick[-1]+5)
            if i == 0:
                legends = [
                    '#$B_S$:{0}'.format(n_s),
                    '#$B_S$:{0}'.format(n_s),
                ]
            else:
                legends = None

            title = None
            _, plots = plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                             xtick_labels, ytick, ytick_labels, title, ax, fig, \
                             ylim, xlim, legends, xtickRotate=0, \
                             linestyles=[linestyles.pop()]*len(ys), cs=cs)
            text = '{0} $\\Rightarrow$ {1}'.format(\
                    anon_building_dict[source],
                    anon_building_dict[target])
            ax.text(0.8, 0.1, text, transform=ax.transAxes, ha='right',
                    backgroundcolor='white'
                    )#, alpha=0)
            plot_list += plots

    axes[0].legend(bbox_to_anchor=(0.15, 0.96), ncol=3, frameon=False)
    for ax in axes:
        ax.grid(True)
    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    plt.text(0, 1.16, '$F_1$: \nMacro $F_1$: ', va='center', ha='center', 
            transform=axes[0].transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center')

    save_fig(fig, 'figs/crf.pdf')
    subprocess.call('./send_figures')


def crf_entity_result():
    building_sets = [('ebu3b', 'ap_m'), ('ap_m', 'bml'),
                 ('ebu3b', 'ghc'), ('ghc', 'ebu3b'), ('ebu3b', 'bml', 'ap_m')] ### TODO TODO: this should be changed to use ebu3b,ap_m -> bml
    #building_sets = [('ebu3b', 'ghc'), ('ebu3b', 'ghc')]
    #building_sets = [('ap_m',), ('bml',),
    #             ('ghc',), ('ebu3b',), ('ap_m',)] ### TODO TODO: this should be changed to use ebu3b,ap_m -> bml
    fig, axes = plt.subplots(1, len(building_sets))
    with open('result/baseline.json', 'r') as fp:
        baseline_results = json.load(fp)

    cs = ['firebrick', 'deepskyblue']
    plot_list = list()
    acc_better_list = []
    mf1_better_list = []
    comp_xs = [10, 50, 150]
    for i, (ax, buildings) in enumerate(zip(axes, building_sets)):
        print(i)
        # Config
        ylim = (-2, 105)
        xlim = (-2, 205)

        # Baseline with source
        result = baseline_results[str(buildings)]
        init_ns = result['ns']
        sample_numbers = result['sample_numbers']
        baseline_acc = result['avg_acc']
        std_acc = result['std_acc']
        baseline_mf1 = result['avg_mf1']
        std_mf1 = result['std_mf1']
        xlabel = '# Target Building Samples'
        ys = [baseline_acc, baseline_mf1]
        baseline_x = sample_numbers
        #xtick = sample_numbers
        #xtick_labels = [str(no) for no in sample_numbers]
        #xtick = [0] + [5] + xtick[1:]
        xtick = [10] + list(range(40, 205, 40))
        #xtick = list(range(0, 205, 40))
        xtick_labels = [str(n) for n in xtick]
        ytick = list(range(0, 105, 20))
        ytick_labels = [str(no) for no in ytick]
        ylabel = 'Score (%)'
        ylabel_flag = False
        linestyles = [':', ':']
        if i == 2:
            data_labels = ['Baseline Acc w/ $B_s$', 
                           'Baseline M-$F_1$ w/ $B_s$']
        else:
            data_labels = None
        title = anon_building_dict[buildings[0]]
        for building in  buildings[1:-1]:
            title += ',{0}'.format(anon_building_dict[building])
        title += '$\\Rightarrow${0}'.format(anon_building_dict[buildings[-1]])
        lw = 1.2
        _, plot = plotter.plot_multiple_2dline(baseline_x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)

        # Baseline without source
        result = baseline_results[str((list(buildings)[-1],))]
        init_ns = result['ns']
        sample_numbers = result['sample_numbers']
        avg_acc = result['avg_acc']
        std_acc = result['std_acc']
        avg_mf1 = result['avg_mf1']
        std_mf1 = result['std_mf1']
        xlabel = '# Target Building Samples'
        ys = [avg_acc, avg_mf1]
        x = sample_numbers
        #xtick = sample_numbers
        #xtick_labels = [str(no) for no in sample_numbers]
        #xtick = list(range(0, 205, 40))
        #xtick_labels = [str(n) for n in xtick]
        ytick = list(range(0, 105, 20))
        ytick_labels = [str(no) for no in ytick]
        ylabel = 'Score (%)'
        ylabel_flag = False
        linestyles = ['-.', '-.']
        if i == 2:
            data_labels = ['Baseline Acc w/o $B_s$', 
                           'Baseline M-$F_1$ w/o $B_s$']
        else:
            data_labels = None
        title = anon_building_dict[buildings[0]]
        for building in  buildings[1:-1]:
            title += ',{0}'.format(anon_building_dict[building])
        title += '$\\Rightarrow${0}'.format(anon_building_dict[buildings[-1]])
        lw = 1.2
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)
        
        # Scrabble without source
        buildingfix = ''.join([buildings[-1]] * 2)
        filename = 'result/crf_entity_iter_{0}_char2tagset_iter_nosource1.json'\
                       .format(buildingfix)
        if not os.path.exists(filename):
            continue
        with open(filename, 'r') as fp:
            res = json.load(fp)
        source_num = 0
        srcid_lens = [len(r['learning_srcids']) - source_num for r in res]
        accuracy = [r['result']['entity']['accuracy'] * 100 for r in res]
        mf1s = [r['result']['entity']['macro_f1'] * 100 for r in res]
        x = srcid_lens
        ys = [accuracy, mf1s]
        linestyles = ['--', '--']
        if i == 2:
            data_labels = ['Scrabble Acc w/o $B_s$', 
                           'Scrabble M-$F_1$ w/o $B_s$']
        else:
            data_labels = None
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)
        """
        filename_template = 'result/crf_entity_iter_{0}_char2tagset_iter_nosource{1}.json'
        nosource_x = range(10, 205, 10)                                         
        x_cands = []                                                            
        acc_cands = []                                                         
        mf1_cands = []
        for exp_num in range(0, 3):                                             
            filename = filename_template.format(buildingfix, exp_num)           
            if not os.path.exists(filename):                                    
                continue                                                        
            with open(filename, 'r') as fp:                                     
                res = json.load(fp)                                             
            source_num = 200 * (len(buildings) - 1)                             
            x_cand = [len(r['learning_srcids']) - source_num for r in res]         
            acc_cand = [r['result']['entity']['accuracy'] * 100 for r in res]   
            mf1_cand = [r['result']['entity']['macro_f1'] * 100 for r in res]   
            x_cands.append(x_cand)                                              
            acc_cands.append(acc_cand)                                          
            mf1_cands.append(mf1_cand)                                          
        nosource_acc = lin_interpolated_avg(nosource_x, x_cands, acc_cands)     
        nosource_mf1 = lin_interpolated_avg(nosource_x, x_cands, mf1_cands)     
        ys = [nosource_acc, nosource_mf1]                                       
            
            
        linestyles = ['--', '--']                                               
        _, plot = plotter.plot_multiple_2dline(nosource_x, ys, xlabel, ylabel, xtick,
                xtick_labels, ytick, ytick_labels, title,             
                ax, fig, ylim, xlim, data_labels, 0, linestyles,   
                cs, lw)
        """

        # Scrabble with source
        buildingfix = ''.join(list(buildings) + [buildings[-1]])

        filename_template = 'result/crf_entity_iter_{0}_char2tagset_iter_{1}.json'
        x = range(10, 205, 10)
        x_cands = []
        acc_cands = []
        mf1_cands = []
        for exp_num in range(0, 3):
            filename = filename_template.format(buildingfix, exp_num)
            if not os.path.exists(filename):
                continue
            with open(filename, 'r') as fp:
                res = json.load(fp)
            source_num = 200 * (len(buildings) - 1)
            x_cand = [len(r['learning_srcids']) - source_num for r in res]
            acc_cand = [r['result']['entity']['accuracy'] * 100 for r in res]
            mf1_cand = [r['result']['entity']['macro_f1'] * 100 for r in res]
            x_cands.append(x_cand)
            acc_cands.append(acc_cand)
            mf1_cands.append(mf1_cand)
        acc = lin_interpolated_avg(x, x_cands, acc_cands)
        mf1 = lin_interpolated_avg(x, x_cands, mf1_cands)
        ys = [acc, mf1]
        
        print(buildings)
        mf1_betters = []
        acc_betters = []
        for comp_x in comp_xs:
            try:
                comp_idx_target = x.index(comp_x)
                comp_idx_baseline = baseline_x.index(comp_x)
                acc_better = \
                    acc[comp_idx_target]/baseline_acc[comp_idx_baseline] - 1
                mf1_better = \
                    mf1[comp_idx_target]/baseline_mf1[comp_idx_baseline] - 1
                """
                acc_better = \
                    acc[comp_idx_target] - baseline_acc[comp_idx_baseline] - 1
                mf1_better = \
                    mf1[comp_idx_target] - baseline_mf1[comp_idx_baseline] - 1
                """
                mf1_betters.append(mf1_better)
                acc_betters.append(acc_better)
                print('srouce#: {0}'.format(comp_x))
                print('Acc\t baseline: {0}\t scrbl: {1}\t better: {2}\t'
                      .format(
                          baseline_acc[comp_idx_baseline],
                          acc[comp_idx_target], 
                          acc_better
                          ))
                print('MF1\t baseline: {0}\t scrbl: {1}\t better: {2}\t'
                      .format(
                          baseline_mf1[comp_idx_baseline],
                          mf1[comp_idx_target], 
                          mf1_better
                          ))
            except:
                pdb.set_trace()
        mf1_better_list.append(mf1_betters)
        acc_better_list.append(acc_betters)

        linestyles = ['-', '-']
        if i == 2:
            data_labels = ['Scrabble Acc w/ $B_s$', 
                           'Scrabble M-$F_1$ w/ $B_s$']
        else:
            data_labels = None
        _, plot = plotter.plot_multiple_2dline(x, ys, xlabel, ylabel, xtick,
                             xtick_labels, ytick, ytick_labels, title,
                             ax, fig, ylim, xlim, data_labels, 0, linestyles,
                                               cs, lw)
        plot_list.append(plot)

        if i == 2:
            ax.legend(bbox_to_anchor=(3.5, 1.53), ncol=4, frameon=False)
            #ax.legend(bbox_to_anchor=(3.2, 1.45), ncol=4, frameon=False)
    print('====================')
    print('Source nums: {0}'.format(comp_xs))
#    pdb.set_trace()
    mf1_better_avgs = [np.mean(list(map(itemgetter(i), mf1_better_list)))
                       for i, _ in enumerate(comp_xs)]
    acc_better_avgs = [np.mean(list(map(itemgetter(i), acc_better_list)))
                       for i, _ in enumerate(comp_xs)]
    print('MF1 better in average, {0}'.format(mf1_better_avgs))
    print('Acc better in average, {0}'.format(acc_better_avgs))
    

    fig.set_size_inches(9, 1.5)
    for ax in axes:
        ax.grid(True)
    for i in range(1,len(building_sets)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')
    for i in range(0,len(building_sets)):
        if i != 2:
            axes[i].set_xlabel('')

    #legends_list = ['Baseline A', 'Baseline MF']
    #axes[2].legend(loc='best', legends_list)


    save_fig(fig, 'figs/crf_entity.pdf')
    subprocess.call('./send_figures')
        
def word_sim_comp():
    buildings = ['ebu3b', 'ap_m', 'bml', 'ghc']
    word_sim_dict = dict()
    token_sim_dict = dict()
    adder = lambda x,y: x + y
    for b1 in buildings:
        for b2 in buildings:
            with open('metadata/{0}_char_sentence_dict_justseparate.json'.format(b1), 'r') as fp:
                b1_s_dict = json.load(fp)
            with open('metadata/{0}_char_sentence_dict_justseparate.json'.format(b2), 'r') as fp:
                b2_s_dict = json.load(fp)
            b1_words = set(reduce(adder, [re.findall('[a-zA-Z]+', ''.join(s)) for s in b1_s_dict.values()]))
            b2_words = set(reduce(adder, [re.findall('[a-zA-Z]+', ''.join(s)) for s in b2_s_dict.values()]))
            word_sim_dict['#'.join([b1, b2])] = len(b1_words.intersection(b2_words)) / \
                                          len(b2_words)
            with open('metadata/{0}_sentence_dict_justseparate.json'.format(b1), 'r') as fp:
                b1_s_dict = json.load(fp)
            with open('metadata/{0}_sentence_dict_justseparate.json'.format(b2), 'r') as fp:
                b2_s_dict = json.load(fp)
            b1_tokens = set([s for s in reduce(adder, b1_s_dict.values()) if s.isalpha()])
            b2_tokens = set([s for s in reduce(adder, b2_s_dict.values()) if s.isalpha()])
            token_sim_dict['#'.join([b1, b2])] = len(b1_tokens.intersection(b2_tokens)) / \
                                          len(b2_tokens)
    with open('result/word_sim.json', 'w') as fp:
        json.dump(word_sim_dict, fp)
    with open('result/token_sim.json', 'w') as fp:
        json.dump(token_sim_dict, fp)

    for b1 in buildings:
        for b2 in buildings:
            with open('metadata/{0}_char_sentence_dict_justseparate.json'.format(b1), 'r') as fp:
                b1_s_dict = json.load(fp)
            with open('metadata/{0}_char_sentence_dict_justseparate.json'.format(b2), 'r') as fp:
                b2_s_dict = json.load(fp)
            b1_words = set(reduce(adder, [re.findall('[a-zA-Z]+', ''.join(s)) for s in b1_s_dict.values()]))
            b2_words = list(reduce(adder, [re.findall('[a-zA-Z]+', ''.join(s)) for s in b2_s_dict.values()]))
            word_sim_dict['#'.join([b1, b2])] = len([1 for w in b2_words if w in b1_words]) / \
                                          len(b2_words)
            with open('metadata/{0}_sentence_dict_justseparate.json'.format(b1), 'r') as fp:
                b1_s_dict = json.load(fp)
            with open('metadata/{0}_sentence_dict_justseparate.json'.format(b2), 'r') as fp:
                b2_s_dict = json.load(fp)
            b1_tokens = set([s for s in reduce(adder, b1_s_dict.values()) if s.isalpha()])
            b2_tokens = list([s for s in reduce(adder, b2_s_dict.values()) if s.isalpha()])
            token_sim_dict['#'.join([b1, b2])] = len([1 for t in b2_tokens if t in b1_tokens]) / \
                                          len(b2_tokens)
    with open('result/word_sim_weighted.json', 'w') as fp:
        json.dump(word_sim_dict, fp)
    with open('result/token_sim_weighted.json', 'w') as fp:
        json.dump(token_sim_dict, fp)


def entity_iter_result():
    source_target_list = [('ebu3b', 'ap_m'),
                          #('ebu3b', 'ap_m'),
                          #('ghc', 'ebu3b')
                          ('ghc', 'ap_m')
                          ]
    ts_flag = False
    eda_flag = False
    fig, axes = plt.subplots(1, len(source_target_list))
#    axes = [ax]
    cs = ['firebrick', 'deepskyblue']
    for i, (ax, (source, target)) in enumerate(zip(axes, source_target_list)):

        #filename_template = 'result/entity_iter_{0}_{1}2.json'
        filename_template = 'result/entity_iter_{0}_{1}{2}.json'
        prefixes = [(''.join([target]*2), 'nosource_nosa'),
                    (''.join([target]*2), 'nosource_sa'),
                    (''.join([source, target, target]), 'source_nosa'),
                    (''.join([source, target, target]), 'source_sa')
                    ]
        linestyles = [':', '--', '-.', '-']
        for linestyle, (buildingfix, optfix) in zip(linestyles, prefixes):
            sa_flag = 'X' if 'nosa' in optfix else 'O'
            src_flag = '0' if 'nosource' in optfix else '200'
            source_num = int(src_flag)
            """
            filename = filename_template.format(buildingfix, optfix)
            if not os.path.exists(filename):
                continue
            with open(filename, 'r') as fp:
                data = json.load(fp)[1:]
            x_t = [len(set(datum['learning_srcids'])) - source_num for datum in data]
            accs = [val * 100 for val in data[-1]['accuracy_history']]
            mf1s = [val * 100 for val in data[-1]['macro_f1_history']]
            ys = [accs, mf1s]
            """
            #if sa_flag == 'X' and src_flag == '0':
            #    pdb.set_trace()
            x_t = range(10,201,10)
            acc_cands = []
            mf1_cands = []
            x_cands = []
            for exp_num in range(1,3):
                filename = filename_template.format(buildingfix, optfix, exp_num)
                if not os.path.exists(filename):
                    continue
                with open(filename, 'r') as fp:
                    #data = json.load(fp)[1:]
                    data = json.load(fp)
                x = [len(set(datum['learning_srcids'])) - source_num for datum in data[:-1]]
                #if optfix == 'nosource_nosa':
                #    pdb.set_trace()
                acc = [val * 100 for val in data[-1]['accuracy_history']]
                mf1 = [val * 100 for val in data[-1]['macro_f1_history']]
                x_cands.append(x)
                acc_cands.append(acc)
                mf1_cands.append(mf1)
            if len(x_cands) == 1:
                pdb.set_trace() # for debugging of not existing enough exp data
            mf1s = lin_interpolated_avg(x_t, x_cands, mf1_cands)
            accs = lin_interpolated_avg(x_t, x_cands, acc_cands)
            ys = [accs, mf1s]
            
            if optfix == 'source_sa':
                pdb.set_trace()

            xlabel = None
            ylabel = 'Score (%)'
            xtick = [10] + list(range(50,205, 50))
            xtick_labels = [str(n) for n in xtick]
            ytick = range(0,102,20)
            ytick_labels = [str(n) for n in ytick]
            ylim = (ytick[0]-1, ytick[-1]+2)
            if i==0:
                legends = [
                    '{0},SA:{1}'
                    .format(src_flag, sa_flag),
                    '{0},SA:{1}'
                    .format(src_flag, sa_flag)
                ]
            else:
                legends = None
            title = None
            plotter.plot_multiple_2dline(x_t, ys, xlabel, ylabel, xtick,\
                             xtick_labels, ytick, ytick_labels, title, ax,\
                             fig, ylim, None, legends, xtickRotate=0, \
                             linestyles=[linestyle]*len(ys), cs=cs)
            if optfix == 'sa_source':
                pdb.set_trace()

    for ax in axes:
        ax.grid(True)
    for ax, (source, target) in zip(axes, source_target_list):
        #ax.set_title('{0} $\Rightarrow$ {1}'.format(
        #    anon_building_dict[source], anon_building_dict[target]))
        #ax.text(0.45, 0.2, '{0} $\Rightarrow$ {1}'.format(
        ax.text(0.45, 0.2, '{0} $\Rightarrow$ {1}'.format(
            anon_building_dict[source], anon_building_dict[target]),
            fontsize=11,
            transform=ax.transAxes,
            #backgroundcolor='white'
            )

    for i in range(1,len(source_target_list)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')

    ax = axes[0]
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [0,1,2,3,4,5,6,7]
    new_handles = [handles[i] for i in legend_order]
    new_labels = [labels[i] for i in legend_order]
    ax.legend(new_handles, new_labels, bbox_to_anchor=(0.15,0.96), ncol=4, 
              frameon=False, handletextpad=0.15, columnspacing=0.7)
    #ax.legend(new_handles, new_labels, bbox_to_anchor=(0.23,1.35), ncol=3, frameon=False)
    plt.text(-0.0, 1.18, 'Accuracy: \nMacro $F_1$: ', ha='center', va='center',
            transform=ax.transAxes)
    fig.text(0.5, -0.1, '# of Target Building Samples', ha='center')

    for i, ax in enumerate(axes):
        if i != 0:
            ax.set_xlabel('')

    fig.set_size_inches(4.4,1.5)
    save_fig(fig, 'figs/entity_iter.pdf')
    subprocess.call('./send_figures')


DEBUG = True 
def finishing_condition_check():
    building_sets = [('ebu3b', 'ap_m'), ('ap_m', 'bml'),
                 ('ebu3b', 'ghc'), ('ghc', 'ebu3b'), ('ebu3b', 'bml', 'ap_m')]
    scrabble_template =  'result/crf_entity_iter_{0}_char2tagset_iter_{1}.json'
    for building_set in building_sets:
        buildingfix = ''.join(list(building_set) + [building_set[-1]])
        with open('metadata/{0}_ground_truth.json'.format(building_set[-1]), 'r') as fp:
            truths_dict = json.load(fp)
        exp_nums = range(1,4)
        print('==============')
        print(building_set)
        for exp_num in exp_nums:
            filename = scrabble_template.format(buildingfix, exp_num)
            if not os.path.exists(filename):
                continue
            with open(filename, 'r') as fp:
                result = json.load(fp)
            srcid_history = [len(set(res['learning_srcids'])) - 200 
                             for res in result]
            print('learning_srcids: ', srcid_history)
            for i, prev_res in enumerate(result[:-1]):
                curr_res = result[i+1]
                prev_srcids = list(set(prev_res['learning_srcids']))
                curr_srcids = list(set(curr_res['learning_srcids']))
                new_srcids = [srcid for srcid in curr_srcids 
                              if srcid not in prev_srcids]
                prev_correct_srcids = prev_res['result']['entity']['correct']
                redundant_new_srcids = [srcid for srcid in new_srcids
                                        if srcid in prev_correct_srcids]
                print(len(redundant_new_srcids) / len(new_srcids))
                if DEBUG and False:
                    for srcid in redundant_new_srcids:
                        print('TRUE!!!!!!!!!!!!!!!!!!!')
                        print(truths_dict[srcid])
                        print('PRED???????????????????')
                        print(prev_res['result']['entity']['correct'][srcid]['tagsets'])
                if DEBUG:
                    for srcid in new_srcids:
                        if srcid in curr_res['result']['entity']['incorrect']:
                            print('TRUE!!!!!!!!!!!!!!!!!!!')
                            print(truths_dict[srcid])
                            print('PRED???????????????????')
                            print(curr_res['result']['entity']['incorrect'][srcid]['tagsets'])
                            pdb.set_trace()
                            
        pdb.set_trace()
                
                
def cls_comp_result():
    source_target_list = ('ebu3b', 'ap_m')
    keys = ['best', 'ts', 'rf']
    xs = list(range(10, 205, 10))
    accuracy_dict = OrderedDict({
        'best': [89.809313820507768, 92.54815950011843, 94.820762260127921, 95.97224073086943, 96.084653841183666, 96.189745940212362, 96.621875740345899, 96.767353707652205, 97.25703698768065, 97.303271588486126, 97.563484660033183, 98.26716491945038, 97.689250918028904, 98.192926735370776, 98.38512052831085, 98.332192527621629, 98.393721664943683, 98.662756406459749, 98.887643256929636, 98.967675573027705],
        'ts': [0.8939772861881065, 0.8923213679976736, 0.9123210382072324, 0.9135980339105342, 0.9189532249466957, 0.9340140813788202, 0.9352186241411988, 0.9355258676853828, 0.9291091215997943, 0.9378608124876789, 0.9319247243221132, 0.949146448493464, 0.9489394545131488, 0.9502468717020965, 0.9567056828950493, 0.9472988480217964, 0.9615234837716184, 0.966066986496091, 0.9657838041933192, 0.9655206112295668],
        'rf':[0.806640902629711,
             0.8715051972281449,
              0.8819351901208243,
               0.8936811478322669,
                0.9154761904761907,
                 0.9102993218431644,
                  0.9163216654821128,
                   0.9187111318407958,
                    0.9251769426676142,
                     0.9323353470741529,
                      0.9335880123193552,
                       0.9353082059938402,
                        0.9385065002369106,
                         0.9440912994551051,
                          0.9449197465055669,
                           0.9479770048566685,
                            0.95334636342099,
                             0.9520936981757874,
                              0.9534481609808099,
                               0.9574993336886989,
                                0.9613235903814261],       
    })
    mf1_dict = OrderedDict({
        'best': [49.278915576009666, 54.796766717693828, 62.58888234797125, 65.516750225788741, 68.292157713216596, 70.178737730933733, 72.269065905342927, 75.530080228774239, 79.910634234930825, 83.958759694464149, 86.604737828403415, 89.944532313205116, 89.509558650993768, 92.646954050881263, 92.840673983293001, 92.748649991145385, 93.127511989870385, 93.479568639265494, 94.246971132932828, 94.718836697647319],
        'ts': [0.56653458779577659, 0.55708814049375366, 0.5937535218897827, 0.63466926766986798, 0.653458865790845, 0.64011173425185053, 0.67281122169885288, 0.68270291522350057, 0.72076990493532245, 0.71261982497230925, 0.70044729648937165, 0.77730251488642088, 0.76286044963642097, 0.79628750932789027, 0.81995259322192149, 0.81512563219291001, 0.83983065742402829, 0.85147624388541865, 0.85183408423723528, 0.85288622740244369],
        'rf': [0.12250376794594604,
             0.18942204544104752,
              0.22171884155985688,
               0.27069328069179505,
                0.30405631973712544,
                 0.2984466141860372,
                  0.3205452968001699,
                   0.33832465365023096,
                    0.3891719868291194,
                     0.44145987155626004,
                      0.4629123930116906,
                       0.4960558419219113,
                        0.5335594108556089,
                         0.5915815154291774,
                          0.6430516639970087,
                           0.6950590411205589,
                            0.7300801879845085,
                             0.7553289202919391,
                              0.7856917033978976,
                               0.8454315647144195,
                                0.8931418245685142],
    })

    for k, v in mf1_dict.items():
        if k == 'best':
            mf1_dict[k] = [vvv/100 for vvv in v[:len(xs)]]
        else:
            mf1_dict[k] = v[:len(xs)]

    for k, v in accuracy_dict.items():
        if k == 'best':
            accuracy_dict[k] = [vvv/100 for vvv in v[:len(xs)]]
        else:
            accuracy_dict[k] = v[:len(xs)]

    legends = ['OCC', 'OCC w/ TS', 'RF'] * 2
    linestyles = ['-', ':', '-.'] * 2
    cs = ['firebrick']*len(keys) + ['deepskyblue'] * len(keys)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(3,1.7)
    axes = [ax]
    mult = lambda x: x*100
    hundreder = lambda seq: list(map(mult, seq))
    ys = list(map(hundreder, list(accuracy_dict.values()) + list(mf1_dict.values())))
    #ys = [char_precs, phrase_f1s, char_macro_f1s, phrase_macro_f1s]
    xlabel = '# of Target Building Samples'
    ylabel = 'Score (%)'
    xtick = [10] + list(range(40, 205, 40))
    xtick_labels = [str(n) for n in xtick]
    ytick = range(0,101,20)
    ytick_labels = [str(n) for n in ytick]
    xlim = (xtick[0]-2, xtick[-1]+5)
    ylim = (ytick[0]-2, ytick[-1]+5)
    title = None
    _, plots = plotter.plot_multiple_2dline(xs, ys, xlabel, ylabel, xtick,\
                            xtick_labels, ytick, ytick_labels, title, ax, fig, \
                            ylim, xlim, None , xtickRotate=0, \
                            linestyles=linestyles, cs=cs)
   #ax.legend(plots, legends, 'upper center', ncol=4
    #legend_order = [0,4,1,5,2,3]
    legend_order = [0,3,1,4,2,5]
    new_handles = [plots[i] for i in legend_order]
    new_legends = [legends[i] for i in legend_order]
    fig.legend(new_handles, new_legends, ncol=3, bbox_to_anchor=(0.15, 1.08, 0.8, 0.095),
               prop={'size':7}, frameon=False )
    for ax in axes:
        ax.grid(True)
    plt.text(0.03, 1.135, 'Accuracy: \nMacro $F_1$: ', ha='center', va='center',
            transform=ax.transAxes, fontsize=7)
    save_fig(fig, 'figs/cls.pdf')
    subprocess.call('./send_figures')




def str2bool(v):
    if v in ['true', 'True']:
        return True
    elif v in ['false', 'False']:
        return False
    else:
        assert(False)

def str2slist(s):
    s.replace(' ', '')
    return s.split(',')

def str2ilist(s):
    s.replace(' ', '')
    return [int(c) for c in s.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.register('type','slist', str2slist)
    parser.register('type','ilist', str2ilist)

    parser.add_argument(choices= ['crf', 'entity', 'crf_entity', 'entity_iter',
                                  'etc', 'entity_ts', 'cls', 'word_sim', 'finalizing',
                                  'crf_acc'],
                        dest = 'exp_type')
    args = parser.parse_args()

    if args.exp_type == 'crf':
        crf_result()
    if args.exp_type == 'crf_acc':
        crf_result_acc()
    elif args.exp_type == 'entity':
        entity_result()
    elif args.exp_type == 'crf_entity':
        crf_entity_result()
    elif args.exp_type == 'entity_iter':
        entity_iter_result()
    elif args.exp_type == 'entity_ts':
        entity_ts_result()
    elif args.exp_type == 'cls':
        cls_comp_result()
    elif args.exp_type == 'etc':
        etc_result()
    elif args.exp_type == 'word_sim':
        word_sim_comp()
    elif args.exp_type == 'finalizing':
        finishing_condition_check()

