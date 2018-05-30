import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import operator
import matplotlib
import matplotlib.colors as col
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from pytz import timezone
from matplotlib import colors as mcolors

#class plotter:
# dataSeries (2-dimensional np.ndarray), figSize (tuple, length=2) -> fig
# each row of dataSeries is one data type.
# Other details should be implemented in inheritor

# NOTE
# Variable details:
# tickRanges: np.arange, contains tick numbers
# tickTags: list of tick name, contains tick names

pst = timezone("US/Pacific")


def save_fig(fig, name, dpi=400):
    pp = PdfPages(name)
    pp.savefig(fig, bbox_inches='tight', pad_inches=0, dpi=dpi)
    pp.close()


################################################### dataSeries (2-dimensional np.ndarray), figSize (tuple, length=2) -> fig
# dataSeries (list of np.ndarray), figSize (tuple, length=2) -> fig
# stackNum starts from 0, which means no stack but just a bar.
# each row of dataSeries is one data type.
# number of stackNum indicates the dats to be stacked.
# e.g., if length of dataSeries is 7, and stack Num is 2,
# dataSeries[0] and dataSereis[1] should be stacked on same bar
# dataSeries[1] and dataSereis[2] should be stacked on same bar
# dataSeries[3] and dataSereis[4] should be stacked on same bar
# Other details should be implemented in inheritor
def plot_multiple_stacked_bars(dataSeries, stackNum, xlabel=None, ylabel=None, xtickRange=None, xtickTag=None, ytickRange=None, ytickTag=None, title=None, stdSeries=None, axis=None, fig=None, clist=None, dataLabels=None, ylim=None, linewidth=0.2, xtickRotate=None, legendLoc='best', hatchSeries=None, eclist=None, oneBlockWidth=0.8, ytickNum=None, totalBlockWidth=0.8):
    barNum = len(dataSeries)/(stackNum+1)
    #oneBlockWidth = float(0.8/float(barNum))
    oneBlockWidth = float(oneBlockWidth/float(barNum))
    originalOneBlockWidth = float(0.8/float(barNum))
    x = np.arange(0,len(dataSeries[0]))
    if axis==None:
        #axis = plt.gca()
        fig, axis = plt.subplots(1,1)
    bars = list()
    colorIdx = 0
    dataLabelIdx = 0
    hatchLabelIdx = 0
    ecolorIdx = 0
    for barIdx in range(0,barNum):
        xpos = x-totalBlockWidth/2.0 + originalOneBlockWidth*barIdx + originalOneBlockWidth/2.0
        if clist:
            color = clist[colorIdx]
            colorIdx += 1
        else:
            color = None
        if dataLabels:
            dataLabel = dataLabels[dataLabelIdx]
            dataLabelIdx += 1
        else:
            dataLabel = None
        if hatchSeries != None:
            hatch = hatchSeries[hatchLabelIdx]
            hatchLabelIdx += 1
        else:
            hatch=None
        if eclist != None:
            ecolor = eclist[ecolorIdx]
            ecolorIdx +=1 
        else:
            ecolor = None
        #bars.append(axis.bar(xpos, dataSeries[barIdx*(stackNum+1)], yerr=std, width = oneBlockWidth, align='center', color=color, label=dataLabel, linewidth=linewidth, hatch=hatch, ecolor=ecolor))
        bars.append(axis.bar(xpos, dataSeries[barIdx*(stackNum+1)], width = oneBlockWidth, align='center', color=color, label=dataLabel, linewidth=linewidth, hatch=hatch))
        if stdSeries:
            std = stdSeries[barIdx*(stackNum+1)]
            axis.errorbar(xpos, dataSeries[barIdx*(stackNum+1)], yerr=std, ecolor=ecolor,elinewidth=0.4, capthick=0.3, fmt=None, capsize=1.3)
        offset = dataSeries[barIdx]
        for stackIdx in range(1,stackNum+1):
            if stdSeries:
                std = stdSeries[barIdx*(stackNum+1)]
            else:
                std = None
            if clist:
                color = clist[colorIdx]
                colorIdx += 1 
            else:
                color = None
            if dataLabels:
                dataLabel = dataLabels[dataLabelIdx]
                dataLabelIdx += 1
            else:
                dataLabel = None
            if hatchSeries != None:
                hatch = hatchSeries[hatchLabelIdx]
                hatchLabelIdx += 1
            else:
                hatch=None
            if eclist != None:
                ecolor = eclist[ecolorIdx]
                ecolorIdx +=1 
            else:
                ecolor = None
            bars.append(axis.bar(xpos, dataSeries[barIdx*(stackNum+1)+stackIdx], yerr=std, width=oneBlockWidth, bottom=offset, align='center', color=color, label=dataLabel, linewidth=linewidth, hatch=hatch, ecolor=ecolor))
            offset += dataSeries[barIdx*(stackNum+1)+stackIdx]
    
    #plt.xlim(x[0]-1,x[len(x)-1]+1)
    axis.set_xlim(x[0]-1,x[len(x)-1]+1)
    if ylim != None:
        axis.set_ylim(ylim)
    if ylabel:
    #   axis.set_ylabel(ylabel, labelpad=-0.5,fontsize='smaller')
        axis.set_ylabel(ylabel, labelpad=0,fontsize=7)
    if xlabel:
        axis.set_xlabel(xlabel, labelpad=-0.5, fontsize=7)
        #axis.set_xlabel(xlabel, labelpad=-0.5, fontsize='smaller')
    
    if dataLabels: 
        axis.legend(handles=bars, fontsize=7, loc=legendLoc)
    if xtickTag != None:
        if not xtickRange:
            xtickRange = np.arange(0,len(dataSeries[0])+1, math.floor(float(len(dataSeries[0])/(len(xtickTag)-1))))
            #xtickRange = np.arange(0,len(xtickTag))
        if xtickRotate == None:
            xtickRotate = 70
        #plt.xticks(xtickRange, xtickTag, fontsize=10, rotation=70)
        axis.set_xticks(xtickRange)
        axis.set_xticklabels(xtickTag, fontsize=7, rotation=xtickRotate)
    if ytickTag != None:
        if ytickRange==None:
            ytickRange = np.arange(0,len(ytickTag))
        #plt.yticks(ytickRange, ytickTag, fontsize=10)
        axis.set_yticks(ytickRange)
        axis.set_yticklabels(ytickTag, fontsize=7)

    if ytickTag==None and ytickNum!=None:
        ytickRange = np.arange(0,max(dataSeries[0])+1, max(dataSeries[0])/ytickNum)
        axis.set_yticks(ytickRange)
    axis.tick_params(axis='both', labelsize=7)

    if title:
        #plt.title(title)
        axis.set_title(title, y=1.08)
    return fig, bars

def plot_up_down_bars(upData, downData, upStd=None, downStd=None, xlabel=None, ylabel=None, title=None, axis=None, fig=None, upColor=None, downColor=None, dataLabels=None, legendLoc='best', upEColor=None, downEColor=None, linewidth=0.2, blockwidth=0.8, xtickRange=None, xtickTag=None, ylim=None):
    if fig==None and axis==None:
        fig, axis = plt.subplots(1,1)
    barNum = len(upData)
    if barNum != len(downData):
        print("data length mismatch")
        return None
    blockWidth = 0.5
    x = np.arange(0,barNum)
    bars = list()
    if dataLabels != None:
        legendUp = dataLabels[0]
        legendDown = dataLabels[1]
    bars.append(axis.bar(x,upData, color=upColor, align='center', label=legendUp, ecolor=upEColor, linewidth=linewidth, width=blockwidth))
    bars.append(axis.bar(x,downData, color=downColor, align='center', label=legendDown, ecolor=downEColor, linewidth=linewidth, width=blockwidth))
    axis.errorbar(x, upData, yerr=upStd, ecolor=upEColor, elinewidth=0.4, capthick=0.3, fmt=None, capsize=2)
    axis.errorbar(x, downData, yerr=downStd, ecolor=downEColor, elinewidth=0.4, capthick=0.3, fmt=None, capsize=2)
    if dataLabels != None:
        axis.legend(handles=bars, fontsize=7, loc=legendLoc)
    if ylabel !=None:
        axis.set_ylabel(ylabel, labelpad=-1, fontsize=7)
    if xlabel!=None:
        axis.set_xlabel(xlabel, labelpad=-1, fontsize=7)
    axis.set_xlim(x[0]-1,x[len(x)-1]+1)
    if xtickTag != None:
        if not xtickRange:
            xtickRange = np.arange(0,len(dataSeries[0])+1, math.floor(float(len(dataSeries[0])/(len(xtickTag)-1))))
            #xtickRange = np.arange(0,len(xtickTag))
        if xtickRotate == None:
            xtickRotate = 70
        #plt.xticks(xtickRange, xtickTag, fontsize=10, rotation=70)
        axis.set_xticks(xtickRange)
        axis.set_xticklabels(xtickTag, fontsize=7, rotation=xtickRotate)
    axis.tick_params(axis='y',labelsize=7)
    if title:
        plt.title(title)
    if ylim != None:
        axis.set_ylim(ylim)
    return fig

def plot_colormap(data, figSizeIn, xlabel, ylabel, cbarlabel, cmapIn, ytickRange, ytickTag, xtickRange=None, xtickTag=None, title=None):
    fig = plt.figure(figsize = figSizeIn)
    plt.pcolor(data, cmap=cmapIn)
    cbar = plt.colorbar()
    cbar.set_label(cbarlabel, labelpad=-0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xtickTag:
        plt.xticks(xtickRange, xtickTag, fontsize=10)

    plt.yticks(ytickRange, ytickTag, fontsize=10)
    plt.tight_layout()
    if title:
        plt.title(title)
    plt.show()
    return fig

def plot_colormap_upgrade(data, figSizeIn, xlabel, ylabel, cbarlabel, cmapIn, ytickRange, ytickTag, xtickRange=None, xtickTag=None, title=None, xmin=None, xmax=None, xgran=None, ymin=None, ymax=None, ygran=None):
    if xmin != None:
        y, x = np.mgrid[slice(ymin, ymax + ygran, ygran),
                slice(xmin, xmax + xgran, xgran)]
        fig = plt.figure(figsize = figSizeIn)
#       plt.pcolor(data, cmap=cmapIn)
        plt.pcolormesh(x, y, data, cmap=cmapIn)
        plt.grid(which='major',axis='both')
        plt.axis([x.min(), x.max(), y.min(), y.max()])
    else:
        plt.pcolor(data, cmap=cmapIn)

    cbar = plt.colorbar()
    cbar.set_label(cbarlabel, labelpad=-0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
#   if xtickTag:
#       plt.xticks(xtickRange, xtickTag, fontsize=10)
#
#   plt.yticks(ytickRange, ytickTag, fontsize=10)
    plt.tight_layout()
    if title:
        plt.title(title)
    plt.show()
    return fig

def plot_timeseries(x, y, xlabel, ylabel, xticks=None, xtickTags=None, yticks=None, ytickTags=None, titles=None, xtickRotate=None, dateFormat=None,color=None, axis=None, fig=None):
    if axis==None:
        fig, axis = plt.subplots(1,1)
    #plt.xticks(rotation='70')
    axis.plot_date(x, y, linestyle='-', marker='None',tz=pst, color=color)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
#   axis.set_ylim([0.9,3.1])
    if xticks:
        axis.set_xticks(xticks)
    if xtickTags:
        axis.set_xticklabels(xtickTags)
    if yticks:
        axis.set_yticks(yticks)
    if ytickTags:
        axis.set_yticklabels(ytickTags)
    if titles:
        axis.set_title(titles)
    if xtickRotate != None:
        xtickLabel = axis.get_xticklabels()
        axis.set_xticklabels(xtickLabel, rotation=xtickRotate, fontsize=7)
    #fig.autofmt_xdate()
    if dateFormat!=None:
        axis.xaxis.set_major_formatter(dateFormat)


    fig.subplots_adjust(hspace=0.4)
    #fig.autofmt_xdate()
    return fig, axis

# x (list of np.array(datetime)), y (list of np.array(number)) -> fig 
def plot_multiple_timeseries(xs, ys, xlabel, ylabel, xticks=None, xtickTags=None, yticks=None, ytickTags=None, titles=None, xtickRotate=None, dateFormat=None,color=None):
    dataNum = len(ys)
    fig, axes = plt.subplots(dataNum)
    for i, axis in enumerate(axes):
        #plt.xticks(rotation='70')
        axis.plot_date(xs[i], ys[i], linestyle='-', marker='None',tz=pst, color=color)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
#       axis.set_ylim([0.9,3.1])
        if xticks:
            axis.set_xticks(xticks[i])
        if xtickTags:
            axis.set_xticklabels(xtickTags[i])
        if yticks:
            axis.set_yticks(yticks[i])
        if ytickTags:
            axis.set_yticklabels(ytickTags[i])
        if titles:
            axis.set_title(titles[i])
        if xtickRotate != None:
            xtickLabel = axis.get_xticklabels()
            axis.set_xticklabels(xtickLabel, rotation=xtickRotate, fontsize=7)
        #fig.autofmt_xdate()
        if dateFormat!=None:
            axis.xaxis.set_major_formatter(dateFormat)


    fig.subplots_adjust(hspace=0.4)
    #fig.autofmt_xdate()
    return fig, axes


def plot_multiple_2dline(x, ys, xlabel=None, ylabel=None, xtick=None, xtickLabel = None, ytick=None, ytickLabel=None, title=None, axis=None, fig=None, ylim=None, xlim=None, dataLabels=None, xtickRotate=0, linestyles=[], cs=[], lw=1):
    dataNum = len(ys)
    if axis==None and fig==None:
        fig, axis = plt.subplots(1,1)
    if not linestyles:
        linestyles = ['-'] * dataNum
    if not cs:
        cs = [None] * dataNum
    dataLabelIdx = 0
    plotList = list()
    for i in range(0,dataNum):
        if dataLabels:
            dataLabel = dataLabels[dataLabelIdx]
            dataLabelIdx += 1
        else:
            dataLabel = None
        plot = axis.plot(x,ys[i], label=dataLabel, color=cs[i], linestyle=linestyles[i], lw=lw)
        plotList += plot
    if dataLabels:
        axis.legend(fontsize=7, loc='best')
    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    if xtick:
        axis.set_xticks(xtick)
    if xtickLabel:
        axis.set_xticklabels(xtickLabel, rotation=xtickRotate)
    if ytick:
        axis.set_yticks(ytick)
    if ytickLabel:
        axis.set_yticklabels(ytickLabel)
    if title:
        axis.set_title(title)
#   if dataLabels: 
#       plt.legend(handles=plotList, fontsize=7)

    return fig, plotList

def errorbar(x, y, xerr=None, yerr=None, xlabel=None, ylabel=None, axis=None, fig=None, title=None):
    if axis==None and fig==None:
        fig, axis = plt.subplots(1,1)
    axis.errorbar(x,y,xerr=xerr, yerr=yerr, fmt='o')
    if title:
        axis.set_title(title)
    return fig

def plot_yy_bar(dataSeries, xlabel=None, ylabel=None, xtickRange=None, xtickTag=None, ytickRange=None, ytickTag=None, title=None, stdSeries=None, axis=None, fig=None, clist=None, dataLabels=None, yerrs=None, ylim=None, linewidth=None):
    pass
    
def make_month_tag(baseTime, endTime=datetime(2015,6,25)):
    monthTags = list()
#   for i in range(0,21):
    while baseTime<endTime:
        monthTags.append(baseTime.strftime('%b-\'%y'))
        baseTime += timedelta(days=31)

    return monthTags
