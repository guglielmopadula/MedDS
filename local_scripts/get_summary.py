
from staticvariables import *
import os 
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import copernicusmarine
from datetime import datetime, timedelta
import sys
from tqdm import trange
import pandas as pd
from matplotlib.gridspec import GridSpec
from datetime import date, datetime
import importlib.util
def module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("miao", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config=module_from_file(sys.argv[1])


if not os.path.exists(config.plot_path+"/validation_plots"):
    os.makedirs(config.plot_path+"/validation_plots")


test_output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)
test_dataset=nc.Dataset(config.download_path+"/"+test_output_filename+".nc")


testing_re_mean=np.load(config.download_path+"/re_testing_mean.npy")


depth=nc.Dataset(config.download_re_path_all+"/"+"static.nc").variables["depth"][:].data
test_dataset=test_dataset.variables[config.variable][:]
pred_mask=test_dataset.mask
test_dataset=test_dataset.data
mean=np.load(config.download_path+"/mean.npy")



lb=np.load(config.download_path+    "/lb.npy")
ub=np.load(config.download_path+"/ub.npy")
mask=np.logical_not(np.load(config.download_re_path+"/mask.npy"))
mask=np.repeat(np.expand_dims(mask,0),(len(mean)),0)

#print(np.sqrt(np.mean(test_data-mean)**2))


#print(np.sqrt(np.mean(test_data-testing_re_mean)**2))





#This handles superficial data
if len(mask.shape)==3:

    mask=np.repeat(np.expand_dims(mask,1),depth.shape[0],1)
    mask[:,1:]=True

if len(pred_mask.shape)==3:
    pred_mask=np.repeat(np.expand_dims(pred_mask,1),depth.shape[0],1)
    pred_mask[:,1:]=True
    test_dataset=np.repeat(np.expand_dims(test_dataset,1),depth.shape[0],1)
    test_dataset[:,1:]=np.nan



data_all=test_dataset
true_all=np.zeros(mask.shape)
mean_all=np.zeros(mask.shape)
re_all=np.zeros(mask.shape)
re_all[np.logical_not(mask)]=testing_re_mean.reshape(-1)
lb_all=np.zeros(mask.shape)
ub_all=np.zeros(mask.shape)
mean_all[np.logical_not(mask)]=mean.reshape(-1)
lb_all[np.logical_not(mask)]=lb.reshape(-1)
ub_all[np.logical_not(mask)]=ub.reshape(-1)
mean_all[mask]=np.nan
ub_all[mask]=np.nan
lb_all[mask]=np.nan
data_all[pred_mask]=np.nan

#mask[:,1:]=True
#pred_mask[:,1:]=True



mask_intersected=np.logical_not(mask)*np.logical_not(pred_mask)

test_data=test_dataset[mask_intersected].reshape(config.target_prediction_time,-1)
testing_re_mean=re_all[mask_intersected].reshape(config.target_prediction_time,-1)
mean_all=mean_all[mask_intersected].reshape(config.target_prediction_time,-1)

with open(config.plot_path+"/validation_plots/output.txt", "w") as file:
    print("BIAS RE", np.mean((testing_re_mean-test_data)),file=file)
    print("BIAS MODEL", np.mean((mean_all-test_data)),file=file)
    print("RMSE RE", np.sqrt(np.mean((testing_re_mean-test_data)**2)),file=file)
    print("RMSE MODEL", np.sqrt(np.mean((mean_all-test_data)**2)),file=file)
    print("CORR RE", np.corrcoef(testing_re_mean.reshape(-1),test_data.reshape(-1))[0,1],file=file)
    print("CORR MODEL", np.corrcoef(mean_all.reshape(-1),test_data.reshape(-1))[0,1],file=file)
    print("ACC MODEL", np.corrcoef(mean_all.reshape(-1)-testing_re_mean.reshape(-1),test_data.reshape(-1)-testing_re_mean.reshape(-1))[0,1],file=file)

