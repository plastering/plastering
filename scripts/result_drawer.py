import pdb
import json
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 10
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 10
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes

import numpy as np

import plotter
from plotter import save_fig

building_anon_map = {
    'ebu3b': 'A-1',
    'uva_cse': 'B-1',
    'sdh': 'C-1',
    'ghc': 'D-1'
}
colors = ['firebrick', 'deepskyblue']
inferencer_names = ['zodiac', 'al_hong', 'scrabble']
EXP_NUM = 2
LINESTYLES = ['--', '-.', '-']
FIG_DIR = './figs'

def average_data(xs, ys, target_x):
    target_y = np.zeros((1, len(target_x)))
    for x, y in zip(xs, ys):
        yinterp = np.interp(target_x, x, y)
        target_y += yinterp / len(ys) * 100
    return target_y.tolist()[0]

def plot_pointonly_notransfer():
    buildings = ['ebu3b', 'uva_cse', 'sdh', 'ghc']
    #buildings = ['sdh', 'ebu3b']
    outputfile = FIG_DIR + '/pointonly_notransfer.pdf'

    fig, axes = plt.subplots(1, len(buildings))
    xticks = [0, 10] + list(range(50, 251, 50))
    xticks_labels = [''] + [str(n) for n in xticks[1:]]
    yticks = range(0,101,20)
    yticks_labels = [str(n) for n in yticks]
    xlim = (0, xticks[-1])
    #xlim = (-5, xticks[-1]+5)
    ylim = (yticks[0], yticks[-1])
    interp_x = list(range(10, 250, 5))
    for ax_num, (ax, building) in enumerate(zip(axes, buildings)): # subfigure per building
        xlabel = '# of Samples'
        ylabel = 'Metric (%)'
        title = building_anon_map[building]
        linestyles = deepcopy(LINESTYLES)
        for inferencer_name in inferencer_names:
            if building == 'uva_cse' and inferencer_name == 'scrabble':
                continue
            xs = []
            ys = []
            xss = []
            f1s = []
            mf1s = []
            for i in range(0, EXP_NUM):
                with open('result/pointonly_notransfer_{0}_{1}_{2}.json'
                          .format(inferencer_name, building, i)) as  fp:
                    data = json.load(fp)
                xss.append([datum['learning_srcids'] for datum in data])
                if inferencer_name == 'al_hong':
                    f1s.append([datum['metrics']['f1_micro'] for datum in data])
                    mf1s.append([datum['metrics']['f1_macro'] for datum in data])
                else:
                    f1s.append([datum['metrics']['f1'] for datum in data])
                    mf1s.append([datum['metrics']['macrof1'] for datum in data])
            xs = xss[0] # Assuming all xss are same.
            f1 = average_data(xss, f1s, interp_x)
            mf1 = average_data(xss, mf1s, interp_x)
            x = interp_x
            ys = [f1, mf1]
            if ax_num == 0:
                legends = ['MicroF1, {0}'.format(inferencer_name),
                           'MacroF1, {0}'.format(inferencer_name)
                           ]
            else:
                #data_labels = None
                legends = None
            xtickRotate = 45

            _, plots = plotter.plot_multiple_2dline(
                x, ys, xlabel, ylabel, xticks, xticks_labels,
                yticks, yticks_labels, title, ax, fig, ylim, xlim, legends,
                linestyles=[linestyles.pop()]*len(ys), cs=colors,
                xtickRotate=xtickRotate)
    for ax in axes:
        ax.grid(True)
    for i in range(1,len(buildings)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')
    for i in range(0,len(buildings)):
        ax = axes[i]
        ax.tick_params(axis='x', pad=-1.5)
        if i != 1:
            ax.set_xlabel('')
        else:
            ax.xaxis.set_label_coords(1.1, -0.2)

    axes[0].legend(bbox_to_anchor=(4.3, 1.5), ncol=3, frameon=False)
    #axes[0].legend(bbox_to_anchor=(3.2, 1.5), ncol=3, frameon=False)
    fig.set_size_inches((8,2))
    save_fig(fig, outputfile)

def plot_pointonly_transfer():
    buildings = ['ebu3b', 'uva_cse']
    outputfile = FIG_DIR + '/pointonly_transfer.pdf'

    fig, axes = plt.subplots(1, len(buildings))
    xticks = [0, 10] + list(range(50, 251, 50))
    xticks_labels = [''] + [str(n) for n in xticks[1:]]
    yticks = range(0,101,20)
    yticks_labels = [str(n) for n in yticks]
    xlim = (-5, xticks[-1]+5)
    ylim = (yticks[0]-2, yticks[-1]+5)
    interp_x = list(range(10, 250, 5))
    for i, (ax, building) in enumerate(zip(axes, buildings)): # subfigure per building
        xlabel = '# of Samples'
        ylabel = 'Metric (%)'
        title = building_anon_map[building]
        linestyles = deepcopy(LINESTYLES)
        for inferencer_name in inferencer_names:
            xs = []
            ys = []
            xss = []
            f1s = []
            mf1s = []
            for i in range(0, EXP_NUM):
                with open('result/pointonly_transfer_{0}_{1}_{2}.json'
                          .format(inferencer_name, building, i)) as  fp:
                    data = json.load(fp)
                xss.append([datum['learning_srcids'] for datum in data])
                if inferencer_name == 'al_hong':
                    f1s.append([datum['metrics']['f1_micro'] for datum in data])
                    mf1s.append([datum['metrics']['f1_macro'] for datum in data])
                else:
                    f1s.append([datum['metrics']['f1'] for datum in data])
                    mf1s.append([datum['metrics']['macrof1'] for datum in data])
            xs = xss[0] # Assuming all xss are same.
            f1 = average_data(xss, f1s, interp_x)
            mf1 = average_data(xss, mf1s, interp_x)
            x = interp_x
            ys = [f1, mf1]
            if i == 2:
                #data_labels = ['Baseline Acc w/o $B_s$',
                #               'Baseline M-$F_1$ w/o $B_s$']
                legends = ['MicroF1, {0}'.format(inferencer_name),
                           'MacroF1, {0}'.format(inferencer_name)
                           ]
            else:
                #data_labels = None
                legends = None

            _, plots = plotter.plot_multiple_2dline(
                x, ys, xlabel, ylabel, xticks, xticks_labels,
                yticks, yticks_labels, title, ax, fig, ylim, xlim, legends,
                linestyles=[linestyles.pop()]*len(ys), cs=colors)
    save_fig(fig, outputfile)

if __name__ == '__main__':
    plot_pointonly_notransfer()
    #plot_pointonly_transfer()
