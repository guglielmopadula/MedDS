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

if not os.path.exists(config.download_path+"/"+output_filename+".nc"):
    print("Data not downloaded")
    assert False
dataset=nc.Dataset(config.download_path+"/"+output_filename+".nc")
dataset=dataset.variables[config.variable][:]
mask=np.load(config.download_re_path+"/mask.npy")

data=dataset.data[:,mask]
np.save(config.download_path+"/"+"training_data.npy",data)


    
