import numpy as np
import scipy as sp

from scipy import stats
from collections import Counter,defaultdict
from multiprocessing import Pool
# from mongodb_helper import mongodb_helper

'''
for Calbimonte
'''
def get_SS(X, B=2):
    pool = Pool()

    N, D = X.shape

    SS = pool.map(getS_wrapper, [(X[i,:],B) for i in range(N)])
    pool.close() 
    pool.join()
    
    return SS

# initial S (a set of buckets) with first 2*B points 
# with tuple (beg_i, end_i, max_val, min_val, max_val_i, min_val_i)
# S = [(i,i+1,max(ts[i],ts[i+1]),min(ts[i],ts[i+1])) for i in range(0,2*B,2)]
def getS_wrapper(args):
    return getS(*args)

def getS(ts, B):
    S = [(i,i,ts[i],ts[i],i,i) for i in range(0,2*B)]
    for i in range(2*B,len(ts)):
        S.append((i,i,ts[i],ts[i],i,i))
        S = merge_neighbour_buckets(S)
    return S

def merge_neighbour_buckets(S):
    # 2B+1 buckets now, calculate the total error of merging neighbour buckets
    Err = [] # error of merging different buckets
    errs = [get_bucket_err(this_bucket) for this_bucket in S] # errors of each bucket
    max_err = max(errs)
    for i in range(len(S)-1):
        temp_max_err = max_err
        temp_err = get_bucket_err(merge_two_buckets(S[i],S[i+1]))
        temp_max_err = temp_err if temp_err > temp_max_err else max_err
        Err.append(temp_max_err)
    merge_idx = Err.index(min(Err))
    S[merge_idx] = merge_two_buckets(S[merge_idx],S[merge_idx+1])
    S.pop(merge_idx+1)
    return S

def get_bucket_err(s):
        return (s[2]-s[3])/2

def merge_two_buckets(s1,s2):
    s1max = max(s1[2],s2[2]) == s1[2]
    s1min = min(s1[3],s2[3]) == s1[3]
    # get max
    max_val = s1[2] if s1max else s2[2]
    max_val_index = s1[4] if s1max else s2[4]
    # get min
    min_val = s1[3] if s1min else s2[3]
    min_val_index = s1[5] if s1min else s2[5]

    return (min(s1[0],s2[0]),
            max(s1[1],s2[1]),
            max_val, min_val, max_val_index, min_val_index)

def get_piecewise_linear_symbol_feature(slopes,segs=4):
    bins = np.linspace(-np.pi/2,np.pi/2,segs+1)
    symbols = np.digitize(slopes, bins)
    c = Counter(symbols)
    return np.array([c[i+1] for i in range(segs)])

def get_ts_slopes(S):
    return np.array([get_bucket_slope(s) for s in S])

def get_bucket_slope(a):
    return np.arctan((a[2]-a[3])/(a[1]-a[0])*np.sign(a[4]-a[5])) if a[0]!=a[1] else 0


'''for Gao'''
def mode(ndarray,axis=0):
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Attempted to find mode on an empty array!')
    try:
        axis = [i for i in range(ndarray.ndim)][axis]
    except IndexError:
        raise Exception('Axis %i out of range for array with %i dimension(s)' % (axis,ndarray.ndim))
    srt = np.sort(ndarray, axis=axis)
    dif = np.diff(srt, axis=axis)
    shape = [i for i in dif.shape]
    shape[axis] += 2
    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1,-1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)
    bins = np.diff(indices, axis=axis)
    location = np.argmax(bins, axis=axis)
    mesh = np.indices(bins.shape)
    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
    counts = bins[tuple(index)].reshape(location.shape)
    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    
    return (modals, counts)


'''for Hong'''
def get_statF_on_window(X):
    N, D = X.shape
    dim = 11
    F = np.zeros([N, dim])
    # percentiles to be used
    p = [25, 75]

    F[:, 0] = np.min(X, 1)
    F[:, 1] = np.median(X, 1)
    F[:, 2] = np.sqrt(np.mean(np.square(X), 1))
    F[:, 3] = np.max(X, 1)
    F[:, 4] = np.var(X, 1)
    F[:, 5] = sp.stats.skew(X, 1)
    F[:, 6] = sp.stats.kurtosis(X, 1)
    
    # calculate slope
    xx = np.linspace(1, D, D)
    tempx = xx - np.mean(xx)
    F[:, 7] =  tempx.dot( (X-np.mean(X)).T ) / ( tempx.dot(tempx.T) )
    
    # quantiles
    F[:, 8:len(p)+8] = np.vstack([np.percentile(X, i, axis=1) for i in p]).T
    F[:, 10] = F[:, 9] - F[:, 8]

    # check illegal features nan/inf
    F[np.isnan(F)] = 0
    F[np.isinf(F)] = 0

    return F


def window_feature(X,feature_fun,win_num,overlapping=0):
    '''function used to extract features by window sections and concatenate them'''
    if win_num < overlapping:
        print("Error! overlapping length should be smaller than window length")
    N,D = X.shape
    temp = feature_fun(X[:2,:10])
    _,dimf = temp.shape
    F = np.zeros([N,dimf,D//(win_num-overlapping)])
    cnt = 0
    for i in range(0,D-1,win_num-overlapping):
        start = i if i<overlapping else i-overlapping
        temp = feature_fun(X[:,start:start+win_num])
        F[:,:,cnt] = temp
        cnt = cnt + 1
    return F


'''for Balaji'''
def haar_transform(x):
    xc = x.copy()
    n = len(xc)

    avg = [0 for i in range(n)]
    dif = [0 for i in range(n)]

    while n > 1:

        for i in range(int(n/2)):
            avg[i] = (xc[2*i]+xc[2*i+1])/2
            dif[i] = xc[2*i]-avg[i]

        for i in range(int(n/2)):
            xc[i] = avg[i]
            xc[i+int(n/2)] = dif[i]

        n = int(n/2)

    return xc


class data_feature_extractor():

    def __init__(self, X):
        self.X = X
        self.functions = [
        'getF_1994_Li',
        'getF_2012_Calbimonte',
        'getF_2015_Gao',
        'getF_2015_Hong',
        'getF_2015_Bhattacharya',
        'getF_2015_Balaji',
        'getF_2016_Koh'
        ]


    # Feature
    def getF_1994_Li(self):
        ''' 'mean','variance','CV' (coefficient of variation) '''
        X = self.X
        N,D = X.shape
        dim = 3
        F = np.zeros([N,dim])

        F[:,0] = np.mean(X,1)
        F[:,1] = np.var(X,1)
        F[:,2] =  np.std(X,1) / np.mean(X,1)

        names = ['mean','variance','CV']

        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F


    # Feature
    def getF_2012_Calbimonte(self, B=20, segs=5):
        X = self.X

        SS = get_SS(X, B)

        PLSF = np.array([get_piecewise_linear_symbol_feature(get_ts_slopes(S),segs) for S in SS])
        PLSF = PLSF.astype(float)
        
        return PLSF


    # Feature
    def getF_2015_Gao(self):
        ''' 'min','median','mean','max','std','skewness','kurtosis','entropy','percentile'. '''
        X = self.X

        N, D = X.shape
        dim = 15
        F = np.zeros([N, dim])
        # percentiles to be used
        p = [2,9,25,75,91,98]

        F[:, 0] = np.min(X, 1)
        F[:, 1] = np.median(X, 1)
        F[:, 2] = np.mean(X, 1)
        F[:, 3] = np.max(X, 1)
        F[:, 4] = np.std(X, 1)
        F[:, 5] = sp.stats.skew(X, 1)
        F[:, 6] = sp.stats.kurtosis(X, 1)
        
        # digitize the data for the calculation of entropy if it only contains less than 100 discreate values
        XX = np.zeros(X.shape)
        bins = 100
        for i in range(X.shape[0]):
            if len(np.unique(X[i,:])) < bins:
                XX[i,:] = X[i,:]
            else:
                XX[i,:] = np.digitize(X[i,:], np.linspace(min(X[i,:]), max(X[i,:]), num=bins))        
        F[:, 7] = sp.stats.entropy(XX.T)
        
        F[:, 8:len(p)+8] = np.vstack([np.percentile(X,i,axis=1) for i in p]).T
        
        F[:, 14] = mode(X,1)[0]

        names = ['min','median','mean','max','std','skewness','kurtosis',
                 'entropy','p2','p9','p25','p75','p91','p98','mode']

        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F


    # Feature
    def getF_2015_Hong(self):
        X = self.X
        F = window_feature(X, get_statF_on_window, 4, overlapping=2)
        return np.hstack([np.min(F,2), np.max(F,2), np.median(F,2), np.var(F,2)])


    # Feature
    def getF_2015_Bhattacharya(self):
        X = self.X
        mean_var_fun = lambda x: np.vstack([np.mean(x,1), np.var(x,1)]).T
        F = window_feature(X, mean_var_fun, 3, overlapping=0)
        return np.hstack([np.min(F,2), np.max(F,2), np.median(F,2), np.var(F,2)])


    # Feature
    def getF_2015_Balaji(self):
        X = self.X
        N, D = X.shape
        dim = 24
        F = np.zeros([N, dim])
        
        # 1)scale based: mean/max/min/quartiles/range;
        F[:, 0] = np.mean(X, 1)
        F[:, 1] = np.max(X, 1)
        F[:, 2] = np.min(X, 1)
        F[:, 3] = np.percentile(X, 25, axis=1)
        F[:, 4] = np.percentile(X, 75, axis=1)
        F[:, 5] = F[:, 1] - F[:, 2]
        
        # 2)pattern based: 3 Haar wavelets and 3 Fourier coefficients;
        F[:, 6:9] = haar_transform(X)[:, :3] # this does not seem to be right
        F[:, 9:12] = abs(np.fft.fft(X,axis=1)[:, 1:4]) / D # 0-th is the average
        
        # 3)shape based: location and magnitude of top 2 components from piece-wise constant model, error variance;
        F[:, 12:18] = haar_transform(X)[:, 4:10]

        # 4)texture based: first and second var of difference between consecutive samples, max var, 
        # number of up and down changes, edge entropy measure
        F[:, 18] = np.var(np.diff(X,n=1,axis=1), 1) # first difference
        F[:, 19] = np.var(np.diff(X,n=2,axis=1), 1) # second difference
        # max variation??
        F[:, 20] = np.var(X, 1)
        # number of ups
        ct = Counter(np.where(np.diff(X,n=1,axis=1)>0)[0])
        F[:, 21] = [ct[i] for i in range(N)]
        # number of downs
        ct = Counter(np.where(np.diff(X,n=1,axis=1)<0)[0])
        F[:, 22] = [ct[i] for i in range(N)]
        # edge entropy
        # digitize the data for the calculation of entropy if it only contains less than 100 discreate values
        XX = np.zeros(X.shape)
        bins = 100
        for i in range(N):
            if len(np.unique(X[i,:])) < bins:
                XX[i,:] = X[i,:]
            else:
                XX[i,:] = np.digitize(X[i,:], np.linspace(min(X[i,:]),max(X[i,:]),num=bins))        
        F[:, 23] = sp.stats.entropy(XX.T)
        
        
        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F


    # Feature
    def getF_2016_Koh(self):
        X = self.X
        ''' 'mean','var','dominant freq','skewness','kurtosis' '''
        N, D = X.shape
        dim = 7
        F = np.zeros([N,dim])

        F[:, 0] = np.mean(X, 1)
        F[:, 1] = np.var(X, 1)
        F[:, 2] = np.mean(X, 1)
        
        temp_fft = abs(np.fft.fft(X,axis=1)) / D
        F[:, 3:5] = np.vstack([temp_fft[i,:].argsort()[-3:-1][::-1] for i in range(N)])

        F[:, 5] = sp.stats.skew(X, 1)
        F[:, 6] = sp.stats.kurtosis(X, 1)
        
        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F


if __name__ == '__main__':
    #main(sys.argv[1:])
    #insert_into_db("rice_pt_sdh")
    #print get_from_db("rice_pt_soda")
    #insert_timeseries_data("AHU1 Final Filter DP.csv")
    timeseries_helper = timeseries_helper()

    raw_pt = [i.strip().split('\\')[-1].split(',') for i in open('../../data/Rice/pressure/AHU1 Final Filter DP.csv').readlines()]
    X= np.array(raw_pt)
    X.astype(np.float)
    X=np.asfarray(X,float)
    print X
    #print timeseries_helper.getF_2016_Koh(X)

    mdb_helper = mongodb_helper()
    #print mdb_helper.get_points_data("rice_pt_soda")
    #mdb_helper.recursive_file_read("pressure")
    print mdb_helper.get_timeseries_data("2 Mag CHW Return Temp")
    X= np.array(raw_pt)
    X.astype(np.float)
    X=np.asfarray(X,float)
    print X
    print timeseries_helper.getF_2016_Koh(X)
