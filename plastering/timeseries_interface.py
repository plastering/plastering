import pandas as pd
import os

from glob import glob
from arctic import CHUNK_STORE, Arctic
from datetime import datetime as dt

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


def read_from_db(target_building):
    '''
    load the data from for tgt_bldg
    return:
    {
        point name: data
    }
    data is in pandas.DataFrame format with two columns ['date', 'data']
    '''

    print ('loading timeseries data from db for %s...'%target_building)

    conn = Arctic('localhost')
    if target_building not in conn.list_libraries():
        raise ValueError('%s not found in the DB!'%target_building)
    else:
        lib = conn[target_building]
        res = {point: lib.read(point) for point in lib.list_symbols()}
        return res


if __name__ == "__main__":
    '''test'''
    write_wrapper('ucsd','./ucsd/', 1)
    res = read_from_db('ucsd')
    for rr in res:
        print (res)

