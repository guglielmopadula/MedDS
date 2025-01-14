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
start_date = date_obj + timedelta(days=1)
end_date = date_obj + timedelta(days=config.target_prediction_time)
start_date=start_date.strftime("%Y-%m-%d")
end_date=end_date.strftime("%Y-%m-%d")

output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)
copernicusmarine.subset(
dataset_id=dataset_names.format(variables_dict[config.variable][4],variables_dict[config.variable][3]),
variables=[config.variable],
minimum_longitude=basins_dict[config.basin][0],
maximum_longitude=basins_dict[config.basin][1],
minimum_latitude=basins_dict[config.basin][2],
maximum_latitude=basins_dict[config.basin][3],
start_datetime=start_date,
end_datetime=end_date,
output_filename = output_filename,
output_directory = config.download_path,
force_download=True
)

