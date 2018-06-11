import pandas as pd
import os
import pdb

from glob import glob
from arctic import CHUNK_STORE, Arctic
from arctic.date import DateRange
from datetime import datetime as dt
from datetime import date
import arrow


DEFAULT_START_TIME = arrow.get(2017,1,20)
DEFAULT_END_TIME = arrow.get(2017,2,6)

def write_wrapper(target_building, path_to_directory, schema=1):
    '''
    para:
    target_building: the building name and used as library name
    path_to_directory: the path to the directory containing data files
i   schema: schema used in the csv file
    ***only supports csv for now with three different schemas:
    1 - | timestamp(string) | data
    2 - | timestamp(epoch)  | data
    3 - | data column only  |

    return a wrapped iterator for write_to_db
    '''

    os.chdir(path_to_directory)
    files = glob('*.csv')
    points = []
    timestamps = []
    data = []
    for f in files:
        df = pd.read_csv(f)
        tmp = f.split('/')[-1][:-4] #point name, should generalize
        points.append(tmp)

        #generate dateindex from timestamp
        if schema == 3:
            ts = pd.date_range(start=dt.now(), periods=len(df), freq='S')
        elif schema == 2:
            timestamps.append( pd.to_datetime(list(df.iloc[:, 0].astype(float)), unit='s') )
        else:
            timestamps.append( pd.to_datetime(list(df.iloc[:, 0])) )

        #store data
        data.append( df.iloc[:, -1] )

    write_to_db( target_building, zip(points, timestamps, data))


def write_to_db(target_building, iterator):
    '''write the data from a building'''

    conn = Arctic('localhost')

    #create a lib for the tgt_bldg, a lib is akin to a collection
    if target_building not in conn.list_libraries():
        conn.initialize_library(target_building, lib_type=CHUNK_STORE)
        print ('library for %s created'%target_building)

    #connect to the lib for writing
    lib = conn[target_building]

    for sensor, timestamps, data in iterator:
        df = pd.DataFrame({'date': timestamps, 'data': data})
        df.set_index('date')
        lib.write(sensor, df)
        print ('writing %s is done'%sensor)


def read_from_db(target_building, start_time=None, end_time=None):
    '''
    load the data from for tgt_bldg
    return:
    {
        point name: data
    }
    data is in pandas.DataFrame format with two columns ['date', 'data']
    '''
    if isinstance(start_time, arrow.Arrow):
        start_time = start_time.datetime
    elif isinstance(start_time, (dt, date)):
        pass
    elif start_time == None:
        pass
    else:
        raise ValueError('the type of time value is unknown: {0}'
                         .format(type(start_time)))
    if isinstance(end_time, arrow.Arrow):
        end_time = end_time.datetime
    elif isinstance(end_time, (dt, date)):
        pass
    elif end_time == None:
        pass
    else:
        raise ValueError('the type of time value is unknown: {0}'
                         .format(type(end_time)))
    if start_time and end_time:
        date_range = DateRange(start=start_time, end=end_time)
    else:
        date_range = None

    print ('loading timeseries data from db for %s...'%target_building)

    conn = Arctic('localhost')
    if target_building not in conn.list_libraries():
        raise ValueError('%s not found in the DB!'%target_building)
    else:
        lib = conn[target_building]
        srcids = lib.list_symbols()
        res = {}
        for srcid in srcids:
            data = lib.read(srcid, chunk_range=date_range)
            if len(data) == 0:
                print('WARNING: {0} has empty data.'.format(srcid))
                pdb.set_trace()
            res[srcid] = data
        print('correctly done')
        return res


if __name__ == "__main__":
    '''test'''
    write_wrapper('ucsd','./ucsd/', 1)
    res = read_from_db('ucsd')
    for rr in res:
        print (res)

