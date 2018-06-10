import argparse

from plastering.timeseries_inferface import *

'''
-b specifies the building name, which is the name used for storing data in DB
-p specifies the path to the directory for time series data files

this interface requires the following dependency:
    https://github.com/manahl/arctic

and the following lib needs to be installed first:
pip install git+https://github.com/manahl/arctic.git

and then run e.g.,:
python timeserie.init -b uva_csv -p /path_to_uva_cse_data_files/

'''

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=str, dest='building', required=True)
parser.add_argument('-p', type=str, dest='path', required=True)

args = parser.parse_args()
building = args.building
path = args.path
print ('------storing data for %s from %s------'%(building, path))

#write to DB
write_wrapper(building, path)

#test loading from DB
print ('------testing loading function for %s------'%building)
res = read_from_db(building)
for point,data in res.items():
    print ('data for %s loaded with %d entries'%(point, len(data.data)))

