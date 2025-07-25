from staticvariables import *
import os 
import netCDF4 as nc
import numpy as np
from pandas import date_range
import copernicusmarine
from datetime import datetime, timedelta
import pandas as pd
import sys
import importlib.util
def module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("miao", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config=module_from_file(sys.argv[1])


#config=__import__(sys.argv[1])



date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")
new_date = date_obj - timedelta(days=config.n_train-1)
start_training_date=new_date.strftime("%Y-%m-%d")

start_date=datetime.strptime(start_training_date,"%Y-%m-%d")
end_date=datetime.strptime(config.start_date,"%Y-%m-%d")
start_year=start_date.year
end_year=end_date.year

re_mean=np.load(config.download_re_path+"/"+"reanalysis_mean.npy")
training_mean=np.zeros((0,re_mean.shape[1]))

for year in range(start_year,end_year+1):
    if year%4==0:
        training_mean=np.concatenate((training_mean,re_mean))
    else:
        training_mean=np.concatenate((training_mean,np.delete(re_mean,59,0)),axis=0)

long_range=(date_range(datetime.strptime(str(start_year)+"-01-01","%Y-%m-%d"),datetime.strptime(str(end_year)+"-12-31","%Y-%m-%d")))
short_range=pd.Series(date_range(start_date,end_date))
indices = short_range.apply(lambda x: long_range.get_loc(x) if x in short_range.values else None)
indices=indices.to_numpy()
training_mean=training_mean[indices]
np.save(config.download_path+"/"+"re_training_mean.npy",training_mean)

date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")
start_date = date_obj + timedelta(days=1)
end_date = date_obj + timedelta(days=config.target_prediction_time)
start_year=start_date.year
end_year=end_date.year

testing_mean=np.zeros((0,re_mean.shape[1]))
for year in range(start_year,end_year+1):
    if year%4==0:
        testing_mean=np.concatenate((testing_mean,re_mean))
    else:
        testing_mean=np.concatenate((testing_mean,np.delete(re_mean,59,0)),axis=0)
long_range=(date_range(datetime.strptime(str(start_year)+"-01-01","%Y-%m-%d"),datetime.strptime(str(end_year)+"-12-31","%Y-%m-%d")))
short_range=pd.Series(date_range(start_date,end_date))
indices = short_range.apply(lambda x: long_range.get_loc(x) if x in short_range.values else None)
indices=indices.to_numpy()
testing_mean=testing_mean[indices]
np.save(config.download_path+"/"+"re_testing_mean.npy",testing_mean)
