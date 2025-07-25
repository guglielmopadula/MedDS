from staticvariables import *
import os 
import netCDF4 as nc
import numpy as np
import copernicusmarine
import sys
import pandas as pd
from datetime import datetime, timedelta
import importlib.util
def module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("miao", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config=module_from_file(sys.argv[1])

dir_path=config.dir_path
date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")
new_date = date_obj + timedelta(days=1)
new_date=new_date.strftime("%Y-%m-%d")

download_path_prev=config.download_path+"/{}_{}_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.n_train,config.target_prediction_time,config.timestep)
download_path_new=config.download_path+"/{}_{}_{}_{}_{}_{}".format(config.variable,config.basin,new_date,config.n_train,config.target_prediction_time,config.timestep)



output_filename= "{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,new_date)

copernicusmarine.subset(
    dataset_id=dataset_names.format(variables_dict[config.variable][4],variables_dict[config.variable][3]),
    variables=[config.variable],
    minimum_longitude=basins_dict[config.basin][0],
    maximum_longitude=basins_dict[config.basin][1],
    minimum_latitude=basins_dict[config.basin][2],
    maximum_latitude=basins_dict[config.basin][3],
    start_datetime=new_date,
    end_datetime=new_date,
    output_filename = output_filename,
    output_directory = download_path_prev,
    force_download=True
    )


dataset=nc.Dataset(download_path_prev+"/"+output_filename+".nc")
dataset=dataset.variables[config.variable][:][0]
mask=np.load(download_path_prev+"/"+"mask.npy")
data_prev=np.load(download_path_prev+"/"+"training_data.npy")
data=dataset.data[np.logical_not(mask)]
data_prev[:-1]=data_prev[1:]
data_prev[-1]=data

os.rename(download_path_prev,
            download_path_new)

with open(dir_path+"/config.py", 'r') as file:
    content = file.read()

# Replace all occurrences of old_string with new_string
content = content.replace(config.start_date, new_date)

# Open the file for writing (this will overwrite the file)
with open(dir_path+"/config.py", 'w') as file:
    file.write(content)


training_mean=np.load(download_path_new+"/"+"re_training_mean.npy")
re_mean=np.load(config.download_re_path+"/"+"reanalysis_mean.npy")

new_date_tmp=datetime.strptime("2024-"+new_date[5:],"%Y-%m-%d")
new_date=datetime.strptime(new_date,"%Y-%m-%d")


training_mean[:-1]=training_mean[1:]
training_mean[-1]=re_mean[new_date_tmp.timetuple().tm_yday]

new_date_tmp=new_date+timedelta(days=config.target_prediction_time)
new_date_tmp=new_date_tmp.strftime("%Y-%m-%d")


new_date_tmp=datetime.strptime("2024-"+new_date_tmp[5:],"%Y-%m-%d")

testing_mean=np.load(config.download_path+"/"+"re_testing_mean.npy")

testing_mean[:-1]=testing_mean[1:]
testing_mean[-1]=re_mean[new_date_tmp.timetuple().tm_yday]



np.save(download_path_new+"/"+"training_data.npy",data_prev)
np.save(download_path_new+"/"+"re_training_mean.npy",training_mean)
np.save(download_path_new+"/"+"re_testing_mean.npy",testing_mean)

os.remove(download_path_new+"/"+output_filename+".nc")

