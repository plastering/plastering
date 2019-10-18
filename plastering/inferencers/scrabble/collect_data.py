import json
import logging
import pdb

import numpy as np
import pandas as pd
import arrow
from datetime import datetime

from bd3client.CentralService import CentralService
from bd3client.Sensor import Sensor
from bd3client.Timeseries import Timeseries
from building_depot import DataService, BDError

PST = 'US/Pacific'

# Basic configuration
begin_time_bd2_1 = arrow.get(datetime(2017,1,20), PST).datetime
end_time_bd2_1 = arrow.get(datetime(2017,2,8), PST).datetime

#begin_time_bd3 = arrow.get(datetime(2017,2,1), PST).datetime
#end_time_bd3 = arrow.get(datetime(2017,2,20), PST).datetime
begin_time_bd2_2 = arrow.get(datetime(2015,1,10), PST).datetime
end_time_bd2_2 = arrow.get(datetime(2017,5,10), PST).datetime

#building_name_list = ['AP_M']
#building_name_list = ['Music']
#building_name_list = ['AP_M', 'EBU3B', 'SME', 'Music']
building_name_list = ['BML']
basedir = "data"
header = ['value']
index_label="time"

# Logger configuration
logger = logging.getLogger("data_collection_log")
logger.setLevel(logging.INFO)
log_handler = logging.FileHandler('log/data_collection.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logger.addHandler(log_handler)

# BD2-1 Initialization
with open("config/bd2-1config.json", "r") as fp:
    hostname = json.load(fp)['hostname']
with open("config/bd2-1secrets.json", "r") as fp:
    secrets = json.load(fp)
    username = secrets['username']
    apikey = secrets['apikey']
bd2_1ds = DataService(hostname, apikey, username)

# BD2-2 Initialization
with open("config/bd2-2config.json", "r") as fp:
    hostname = json.load(fp)['hostname']
with open("config/bd2-2secrets.json", "r") as fp:
    secrets = json.load(fp)
    username = secrets['username']
    apikey = secrets['apikey']
bd2_2ds = DataService(hostname, apikey, username)

# BD3 Initialization
with open("config/bd3config.json", "r") as fp:
    hostname = json.load(fp)['hostname']
with open("config/bd3secrets.json", "r") as fp:
    secrets = json.load(fp)
    cid = secrets['cid']
    csecret = secrets['csecret']
bd3cs = CentralService(hostname, cid, csecret)
bd3sensor_api = Sensor(bd3cs)
bd3ts_api = Timeseries(bd3cs)


#Load basic metadata
with open("metadata/bacnet_devices.json", "r") as fp:
    bacnet_devices = json.load(fp)
with open("metadata/building_info.json", "r") as fp:
    building_dict = json.load(fp)

def resample_data(raw_data, begin_time, end_time, sample_method):
    raw_data = raw_data[begin_time:end_time]
    if not begin_time in raw_data.index:
        raw_data[arrow.get(begin_time).to('UTC').datetime] = raw_data.head(1)[0]
    if not end_time in raw_data.index:
        raw_data[arrow.get(end_time).to('UTC').datetime] = raw_data.tail(1)[0]
    raw_data = raw_data.sort_index()
    if sample_method == 'raw':
        proc_data = raw_data
        pass
    elif sample_method == 'nextval':
        proc_data = raw_data.resample('3Min', how='pad')
    else:
        logger.error("sample method not defined well: {0}".format(sample_method))
        assert(False)
    return proc_data


def get_data_bd2(begin_time, end_time, srcid, bd2ds):
    # Get UUID
    """
    try:
        uuid = bd2ds.list_sensors({'source_identifier':srcid})\
                ['sensors'][0]['uuid']
    except:
        logger.error("UUID not found ({0})".format(srcid))
        return None
    """
    uuid = bd2ds.list_sensors({'source_identifier':srcid})\
            ['sensors'][0]['uuid']

    # Get data
    """
    try:
        raw_data = bd2ds.get_timeseries_datapoints(uuid, 'PresentValue', \
                begin_time, end_time)
    except:
        logger.error("Data not found ({0})".format(srcid))
        return None
    """
    raw_data = bd2ds.get_timeseries_datapoints(uuid, 'PresentValue', \
            begin_time, end_time)
#    pdb.set_trace()
    time_list = list()
    value_list = list()
    for row in raw_data['timeseries']:
        for t, v in row.items():
            time_str = t
            value = v
            break
        from dateutil import tz
        time_list.append(arrow.get(time_str).to('UTC').datetime)
        value_list.append(value)
    return pd.Series(index=time_list, data=value_list)
    

def get_data_bd3(begin_time, end_time, srcid):
    # Get UUID
#    try:
    print(srcid)
    (nae, data_type, instance_num)  = srcid.split("_")
    tag_val_dict = {
            "NAENum": nae,
            "BACNet_DataType": data_type,
            "BANet_InstanceNumber": instance_num
            }
    uuid = bd3sensor_api.search(tag_val_dict)['result'][0]['name']
#    except:
#        logger.error("UUID NOT FOUND ({0}, {1})".format(building_name, srcid))
#        return None
    # Get data
#    try:
    ts = bd3ts_api.getTimeseriesDataPoints(uuid, begin_time, end_time)
#    except:
#        logger.error("Data not found ({0})".format(srcid))
    time_list = [arrow.get(row[0]).to('UTC').datetime for row in ts['data']['series'][0]\
            ['values']]
    value_list = [row[2] for row in ts['data']['series'][0]['values']]
    return pd.Series(index=time_list, data=value_list)
    #.to_csv(filename, header=header, index_label=index_label)

def get_data(building_name, begin_time, end_time, srcid):
    if building_name.lower() == "ebu3b":
        return get_data_bd2(begin_time, end_time, srcid, bd2_1ds)
    else:
        return get_data_bd2(begin_time, end_time, srcid, bd2_2ds)
        #return get_data_bd3(begin_time, end_time, srcid)

if __name__ == "__main__":
    cnt = 0
    # Store all data
    for building_name in building_name_list:
        cnt += 1
        if cnt%500==0:
            logger.info("Data download success ({0}, {1})"\
                    .format(building_name, cnt))
        with open("metadata/%s_sentence_dict_justseparate.json"%building_name.lower(), "r") \
                as fp:
            sentence_dict = json.load(fp)
        srcid_list = sentence_dict.keys()
        for srcid in srcid_list:
            if building_name.lower() == 'ebu3b':
                begin_time = begin_time_bd2_1
                end_time = end_time_bd2_1
            else:
                begin_time = begin_time_bd2_2
                end_time = end_time_bd2_2

            ts_series = get_data(building_name, begin_time, end_time, srcid)
            if isinstance(ts_series, pd.Series):
                if len(ts_series)==0:
                    logger.error("Data is empty ({0})".format(srcid))
                    continue
                #ts_series = resample_data(ts_series, begin_time, end_time, \
                #        "raw")
                filename = basedir + '/' + srcid + '.csv'
                ts_series.to_csv(filename, \
                        header=header, \
                        index_label=index_label)
            else:
                logger.error("Data not found ({0})".format(srcid))
