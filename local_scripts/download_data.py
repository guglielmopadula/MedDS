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

date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")
new_date = date_obj - timedelta(days=config.n_train-1)
start_training_date=new_date.strftime("%Y-%m-%d")

output_filename="{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.n_train)
if not os.path.exists(config.download_path):
    os.makedirs(config.download_path)


if not os.path.exists(config.download_path+"/training_data.npy"):
    
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
    force_download=True,
    chunk_size_limit=100
    )
    
else:
    print("Data already downloaded, make sure it is not corrupted before continuing")

    
