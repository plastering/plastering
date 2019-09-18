import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from scipy.fftpack import fft
import dtw
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from scipy.interpolate import interp1d
from scipy import interpolate
#PiecewisePolynomial, 
from six.moves import xrange
#import scipy
import numpy as np
from scipy.interpolate import splrep, splev
import pickle
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from math import sqrt
from copy import deepcopy
from numpy.matlib import repmat
#import FATS
from scipy.signal import savgol_filter 
#import pywt
import pdb

nanAlt = -999999

def normalize(data):
    minVal = min(data)
    minmax = max(data)+1-min(data)
    data = (data-minVal)/minmax
    return data
#       return preprocessing.normalize(data, norm='l2')[0]
#return preprocessing.normalize(data, norm='max')[0]


def get_fft_features(self, inputData, dataDict):
    fftDict = dict()
    idxDict = dict()
    distDict = dict()
    for zone, data in dataDict.iteritems():
        fftDict[zone] = abs(np.fft.rfft(self.normalize(data))[1:10])
        idxDict[zone] = np.where(fftDict[zone]==max(fftDict[zone]))[0][0] 
        
    inputfft = abs(np.fft.rfft(self.normalize(inputData))[1:10])
    inputIdx = np.where(inputfft==max(inputfft))[0][0]
    for zone, data in fftDict.iteritems():
        #               distDict[zone] = np.linalg.norm(fftDict[zone][1:]-inputfft[1:])
        #               distDict[zone] = dtw.dtw(fftDict[zone][1:10],inputfft[1:10])
        distDict[zone] = abs(idxDict[zone]-inputIdx)
        #               print zone, distDict[zone][0]
    return distDict

def get_fft(ts):
    fftResult = fft(ts)
#    print fftResult
    topResults = sorted(fftResult, reverse=True)[0:3]
    result = np.zeros(len(fftResult))
    for top in topResults:
        result[np.where(fftResult==top)] = top
    return result

def get_dominating_freq_range(ts):
    rangeNum = 10
    logInterval = np.log(len(ts)) / rangeNum
    logRange = [logInterval*i for i in range(0,rangeNum)]
    fftResult = fft(ts)
    start = 0
    sumList = list()
    #while start<len(ts):
    for logTick in logRange[1:]:
        start = int(np.exp((logTick-logInterval)))
        end = int(np.exp(logTick))
        sumList.append(sum([abs(val) for val in fftResult[start:end]]))
    maxSum = max(sumList)
    return sumList.index(maxSum)

def fft_coeff(ts):
    fftResult = np.fft.rfft(ts)
    return [abs(fftResult[0]), abs(fftResult[1]), abs(fftResult[2])]

#def wt_coeff(ts):
#    a,b = pywt.dwt(ts, 'db2')
#    return [a[0], a[1], b[0], b[1]]

def get_min(ts):
    return min(ts)

def get_mean(ts):
    return np.mean(ts)

def get_max(ts):
    return max(ts)

def get_amplitude(ts):
    return max(ts) - min(ts)

def get_dtw_features(self, inputData, dataDict):
    dtwDict = dict()
    for zone, data in dataDict.iteritems():
        dtwDict[zone] = dtw.dtw(inputData, data)[0]
    return dtwDict

def get_freq(self, data):
    minmax = max(data) - min(data)
    beforeVal= data[0]
    afterVal = data[0]
    freq = 0
    val1 = data[0]
    val2 = data[0]
    ascendFlag = data[1]>=data[0]
    for datum in data:
        val2 = datum
        if val2==val1:
            ascendFlag = ascendFlag
        else:
            ascendFlag = val2>val1
        if beforeVal>datum and ascendFlag:
            beforeVal = datum
        elif beforeVal<datum and not ascendFlag:
            beforeVal = datum
        if afterVal<datum and ascendFlag:
            afterVal = datum
        elif afterVal>datum and ascendFlag:
            afterVal = datum
            
        if afterVal>=beforeVal+minmax/2 and ascendFlag:
            freq += 0.5
            ascendFlag = not ascendFlag
        elif afterVal<=beforeVal+minmax/2 and not ascendFlag:
            freq += 0.5
            ascendFlag = not ascendFlag
        va1 = datum
                
        return freq

def get_freq_features(self, ts):
    targetFreq = self.get_freq(inputData)
    for zone, data in dataDict.iteritems():
        freqDict[zone] = self.get_freq(data) - targetFreq
    return freqDict

def interp0(test_xs, test_ys, orig_x):
    d_xs = list()
    for x in test_xs[:-1]:
        idx = int(np.where(test_xs==x)[0])
        d_xs.append((test_ys[idx+1] - test_ys[idx])/(test_xs[idx+1]-x))
    plot_y = list()
    for x in orig_x[:-1]:
        idx = np.where(test_xs<=x)[0][-1]
        #       idx = test_xs[(test_xs<=x)]x
        #               plot_y.append((x-test_xs[idx])*d_xs[idx]+test_ys[idx])
        plot_y.append(test_ys[idx])
        plot_y.append(test_ys[-1])
    return plot_y

def interp1(test_xs, test_ys, orig_x):
    inter_f = interp1d(test_xs,test_ys)
    inter_y = inter_f(orig_x)
    return inter_y

#def pla(data, N=20):
def pla(data, period=15):
    N = int(len(data)/period)
    orig_x = range(0,len(data))
    tck = splrep(orig_x, data,s=0)
    test_xs = np.linspace(0,len(data),N)
    spline_ys = splev(test_xs, tck)
    spline_yps = splev(test_xs, tck, der=1)
    xi = np.unique(tck[0])
    yi = [[splev(x, tck, der=j) for j in xrange(3)] for x in xi]
    P = interpolate.PiecewisePolynomial(xi,yi,orders=1)
    test_ys = P(test_xs)
    #inter_y = interp0(test_xs, test_ys, orig_x)
    inter_y = interp1(test_xs, test_ys, orig_x)
    
    mae = sqrt(mean_absolute_error(inter_y, data))
    #       mae = np.var(inter_y-data)
    return mae

#def paa(data, period=15):
def paa(data, period=15):
    numCoeff = int(len(data)/period)
    data = data[:numCoeff*period]
    data = data[:int(len(data)/numCoeff)*numCoeff]
    origData = deepcopy(data)
    N = len(data)
    segLen = int(N/numCoeff)
    sN = np.reshape(data, (numCoeff, segLen))
    g = lambda data: np.mean(data)
    #       avg = np.mean(sN)
    avg = map(g,sN)
    data = np.matlib.repmat(avg, segLen, 1)
    data = data.ravel(order='F')
#       plt.plot(data)
#       plt.plot(origData)
#       plt.show()
#rmse = sqrt(mean_squared_error(data, origData))
    mae = sqrt(mean_absolute_error(data, origData))
#       mae = np.var(origData-data)
    return mae

def get_fats_features(ts):
    fat = FATS.FeatureSpace(Data=['magnitude', 'time'], featureList=['Mean', 'Amplitude', 'Skew', 'Meanvariance', 'PeriodLS', 'Std', 'MaxSlope', 'SmallKurtosis'])
    fat.calculateFeature(npTs)
    return fat.result().tolist()

def get_skew(npts):
    fat = FATS.FeatureSpace(Data=['magnitude', 'time'], featureList=['Skew'])
    fat.calculateFeature(npts)
    result = fat.result().tolist()[0]
    if np.isnan(resulut):
        result = nanAlt
    return result

def get_maxslope(npts):
    return call_fats_func(npts, 'MaxSlope')


def get_periodls(npts):
    return call_fats_func(npts, 'PeriodLS')

def get_smallkurtosis(npts):
    return call_fats_func(npts, 'SmallKurtosis')

def get_fluxpercentilemid20(ts):
    return [call_fats_func(npts, 'FluxPercentileRatioMid20')]

def get_percentile(ts, perc):
    return np.percentile(ts, perc) # 5, 95, 50

def get_percentile20(ts):
    return np.percentile(ts, 20)

def get_percentile80(ts):
    return np.percentile(ts, 80)

def get_lineartrend(npts):
    return call_fats_func(npts, 'LinearTrend')

def get_meanvar(npts):
    return call_fats_func(npts, 'Meanvariance')

def get_std(ts):
    return np.std(ts)

def call_fats_func(npts, featType):
    fat = FATS.FeatureSpace(Data=['magnitude', 'time'], featureList=[featType])
    fat.calculateFeature(npts)
    return fat.result().tolist()[0]

def get_noise_by_sgfilter(data):
    filtered = savgol_filter(data, 9,2)
    diffData = [abs(val1-val2) for val1, val2 in zip(data,filtered)]
    return sum(diffData)

def get_error_rate(ts):
    ts = ts.ravel()
    inds=ts>np.percentile(ts,95)
    inds[0]=True
    inds[-1]=True
    xAll=np.array(range(0,len(ts)))
    f = interp1d(xAll[inds], ts[inds])
    err = ts[:-1]-f(xAll[0:-1])
    return np.sqrt(np.dot(err,err)/len(err))
        

# order of magnitude
def get_oom(ts):
    return np.log(np.mean(ts)+1)

def add_feature(featList, func, ts):
    try:
        result = func(ts)
        if np.isnan(result):
            result = nanAlt
        featList.append(result + 0.01)
    except:
        featList.append(nanAlt)
    return featList

def concat_feature(featList, func, ts):
    try:
        result = func(ts)
        featList = featList + result
    except:
        featList.append(nanAlt)
    return featList

def get_constant():
    return 1

def get_features(ts):
    feats = list()
    normalizedValues = normalize(ts)
    #normalizedValues = ts
    # normalizedNpts = np.asarray([normalizedValues, ts.tolist().index.values])
    feats = add_feature(feats, get_max, ts)
    feats = add_feature(feats, get_min, ts)
    feats = add_feature(feats, get_mean, ts)
    feats = add_feature(feats, get_amplitude, ts)
    feats = add_feature(feats, get_std, ts)
    feats = add_feature(feats, get_error_rate, ts)
    feats = add_feature(feats, get_fft, normalizedValues)
    feats = add_feature(feats, get_dominating_freq_range, normalizedValues)
    
    feats = add_feature(feats, paa, normalizedValues)
    feats = add_feature(feats, pla, normalizedValues)

    #       feats = add_feature(feats, get_smallkurtosis, normalizedNpts)
#       feats = add_feature(feats, get_maxslope, normalizedNpts)
#       feats = add_feature(feats, get_skew, normalizedNpts)
#feats = add_feature(feats, get_fluxpercentilemid20, normalizedNpts)
    get_percentile95 = lambda d:get_percentile(d,95)
    get_percentile5 = lambda d:get_percentile(d,5)
    feats = add_feature(feats, get_percentile95, ts)
    feats = add_feature(feats, get_percentile5, ts)


#    feats = add_feature(feats, get_percentile20, normalizedValues)
#    feats = add_feature(feats, get_percentile80, normalizedValues)
#feats = add_feature(feats, get_lineartrend, normalizedNpts)
#       feats = add_feature(feats, get_meanvar, normalizedNpts)
#    feats = add_feature(feats, get_oom, ts)
    #feats = add_feature(feats, get_fft, normalizedValues)

    feats = add_feature(feats, get_noise_by_sgfilter, normalizedValues) #TODO: Fix this to have significant values
    feats = concat_feature(feats, fft_coeff, normalizedValues) #TODO: Fix this to have significant values
#    feats = concat_feature(feats, wt_coeff, normalizedValues) #TODO: Fix this to have significant values
    #       feats = add_feature(feats, get_periodls, normalizedNpts) # Too slow. Is this necessary?
    
    #       feats = feats + get_fats_features(ts)
    return feats
