from staticvariables import *
import os 
import netCDF4 as nc
import numpy as np
import copernicusmarine
import sys
from datetime import datetime, timedelta
dir_path = os.path.dirname(os.path.realpath(__file__))
config=__import__(sys.argv[1])

date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")
new_date = date_obj + timedelta(days=1)
new_date=new_date.strftime("%Y-%m-%d")

download_path_prev=dir_path+"/"+config.variable+"/"+config.basin+"/{}_{}_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.n_train,config.target_prediction_time,config.timestep)
download_path_new=dir_path+"/"+config.variable+"/"+config.basin+"/{}_{}_{}_{}_{}_{}".format(config.variable,config.basin,new_date,config.n_train,config.target_prediction_time,config.timestep)



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


np.save(download_path_new+"/"+"training_data.npy",data_prev)
os.remove(download_path_new+"/"+output_filename+".nc")

