from staticvariables import *
import os 
import netCDF4 as nc
import numpy as np
import copernicusmarine
from datetime import datetime, timedelta
import sys


config=__import__(sys.argv[1])
if not os.path.exists(config.dir_path+"/"+config.variable):
    os.makedirs(config.dir_path+"/"+config.variable)
if not os.path.exists(config.dir_path+"/"+config.variable+"/"+config.basin):
    os.makedirs(config.dir_path+"/"+config.variable+"/"+config.basin)

date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")
new_date = date_obj - timedelta(days=config.n_train-1)
start_training_date=new_date.strftime("%Y-%m-%d")

output_filename="{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.n_train)
if not os.path.exists(config.download_path):
    os.makedirs(config.download_path)
    copernicusmarine.subset(
    dataset_id=dataset_names.format(variables_dict[config.variable][4],variables_dict[config.variable][3]),
    variables=[config.variable],
    minimum_longitude=basins_dict[config.basin][0],
    maximum_longitude=basins_dict[config.basin][1],
    minimum_latitude=basins_dict[config.basin][2],
    maximum_latitude=basins_dict[config.basin][3],
    start_datetime=start_training_date,
    end_datetime=config.start_date,
    output_filename = output_filename,
    output_directory = config.download_path,
    force_download=True
    )
    
    copernicusmarine.subset(
        dataset_id=static_dataset.format(variables_dict[config.variable][4]),
        minimum_longitude=basins_dict[config.basin][0],
        maximum_longitude=basins_dict[config.basin][1],
        minimum_latitude=basins_dict[config.basin][2],
        maximum_latitude=basins_dict[config.basin][3],
        output_filename = "static",
        output_directory = config.download_path,
        force_download=True
    )
    dataset=nc.Dataset(config.download_path+"/"+output_filename+".nc")
    dataset=dataset.variables[config.variable][:]
    mask=dataset.mask[0]
    data=dataset.data[:,np.logical_not(mask)]
    np.save(config.download_path+"/"+"training_data.npy",data)
    np.save(config.download_path+"/"+"mask.npy",mask)
    os.remove(config.download_path+"/"+output_filename+".nc")
else:
    print("Data already downloaded, make sure it is not corrupted before continuing")

