
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




depth=nc.Dataset(config.download_re_path_all+"/"+"static.nc").variables["depth"][:].data
test_dataset=test_dataset.variables[config.variable][:]
pred_mask=test_dataset.mask
test_dataset=test_dataset.data
mean=np.load(config.download_path+"/mean.npy")

assert False
lb=np.load(config.download_path+"/lb.npy")
ub=np.load(config.download_path+"/ub.npy")

ub=mean+1.96*np.var(mean,axis=0)


mask=np.logical_not(np.load(config.download_re_path+"/mask.npy"))
mask=np.repeat(np.expand_dims(mask,0),(len(mean)),0)

#This handles superficial data
if len(mask.shape)==3:

    mask=np.repeat(np.expand_dims(mask,1),depth.shape[0],1)
    mask[:,1:]=True

if len(pred_mask.shape)==3:
    pred_mask=np.repeat(np.expand_dims(pred_mask,1),depth.shape[0],1)
    pred_mask[:,1:]=True
    test_dataset=np.repeat(np.expand_dims(test_dataset,1),depth.shape[0],1)
    test_dataset[:,1:]=np.nan

true_all=np.zeros(mask.shape)
mean_all=np.zeros(mask.shape)
lb_all=np.zeros(mask.shape)
ub_all=np.zeros(mask.shape)
mean_all[np.logical_not(mask)]=mean.reshape(-1)
lb_all[np.logical_not(mask)]=lb.reshape(-1)
ub_all[np.logical_not(mask)]=ub.reshape(-1)
mean_all[mask]=np.nan
ub_all[mask]=np.nan
lb_all[mask]=np.nan

mask_intersected=np.logical_not(mask)*np.logical_not(pred_mask)
diff=mean_all-test_dataset
diff_uq=(ub_all-test_dataset)*(ub_all<test_dataset)+(lb_all-test_dataset)*(lb_all>test_dataset)


seasons = ['Bias (Autumn, Coast)', 'RMSD (Autumn, Coast)', 
           'Bias-UQ (Autumn, Coast)', 'RMSD-UQ (Autumn, Coast)',
           'Bias (Autumn, Opensea)', 'RMSD (Autumn, Opensea)', 
           'Bias-UQ (Autumn, Opensea)', 'RMSD-UQ (Autumn, Opensea)',
           'Bias (Winter, Coast)', 'RMSD (Winter, Coast)', 
           'Bias-UQ (Winter, Coast)', 'RMSD-UQ (Winter, Coast)',
           'Bias (Winter, Opensea)', 'RMSD (Winter, Opensea)', 
           'Bias-UQ (Winter, Opensea)', 'RMSD-UQ (Winter, Opensea)',
           'Bias (Spring, Coast)', 'RMSD (Spring, Coast)', 
           'Bias-UQ (Spring, Coast)', 'RMSD-UQ (Spring, Coast)',
            'Bias (Spring, Opensea)', 'RMSD (Spring, Opensea)', 
           'Bias-UQ (Spring, Opensea)', 'RMSD-UQ (Spring, Opensea)',
           'Bias (Summer, Coast)', 'RMSD (Summer, Coast)', 'Bias-UQ (Summer, Coast)', 
           'RMSD-UQ (Summer, Coast)',
           'Bias (Summer, Opensea)', 'RMSD (Summer, Opensea)', 'Bias-UQ (Summer, Opensea)', 
           'RMSD-UQ (Summer, Opensea)']
columns = ['0-10 m', '10-30 m', '30-60 m', '60-100 m', '100-150 m']

# Create a DataFrame with random data (you can replace this with actual data)
data = {
    '0-10 m': ["NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN","NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN"],
    '10-30 m': ["NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN","NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN"],
    '30-60 m': ["NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN","NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN"],
    '60-100 m': ["NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN","NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN"],
    '100-150 m': ["NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN","NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN", "NAN"]
}


def get_season(data):
    winter_start_1=date(data.year,1,1)
    winter_end_1=date(data.year,3,31)
    spring_start_1=date(data.year,4,1)
    spring_end_1=date(data.year,6,30)
    summer_start_1=date(data.year,7,1)
    summer_end_1=date(data.year,9,30)
    autumn_start_1=date(data.year,10,1)
    autumn_end_1=date(data.year,12,31)

    if winter_start_1<=data.date()<=winter_end_1:
        return 1
    if spring_start_1<=data.date()<=spring_end_1:
        return 2
    if summer_start_1<=data.date()<=summer_end_1:
        return 3
    if autumn_start_1<=data.date()<=autumn_end_1:
        return 4




# Create the DataFrame
df = pd.DataFrame(data, index=seasons)

# Transpose the DataFrame
df_transposed = df.T

# Create the DataFrame
df = pd.DataFrame(data, index=seasons)
df=df.T

#Copernicus classification of seasons, Winter is from Jan to Apr and summer from Jun to Sept
seasons_names=["Winter","Spring","Summer","Autumn"]


seasons=np.array([1,1,1,2,2,2,3,3,3,4,4,4])
indexes=np.zeros(config.target_prediction_time,dtype=np.int64)
date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")
for i in range(config.target_prediction_time):
    tmp = date_obj + timedelta(days=1+i)
    indexes[i]=get_season(tmp)

start_depth=np.array([0,10,30,60,100])
end_depth=np.array([10,30,60,100,150])
max_depth=depth[np.sum(np.logical_not(mask[0]),axis=0)-1]
coast_index=max_depth<200


for i in trange(len(start_depth)):
    for j in range(1,5):
        tmp_index=np.logical_and(depth>start_depth[i],depth<end_depth[i])
        diff_tmp=diff.copy()
        diff_tmp=diff_tmp[:,:,coast_index]
        diff_tmp=diff_tmp[indexes==j]
        diff_tmp=diff_tmp[:,tmp_index]
        diff_uq_tmp=diff_uq.copy()
        diff_uq_tmp=diff_uq_tmp[:,:,coast_index]
        diff_uq_tmp=diff_uq_tmp[indexes==j]
        diff_uq_tmp=diff_uq_tmp[:,tmp_index]
        mask_tmp=mask_intersected.copy()
        mask_tmp=mask_tmp[:,:,coast_index]
        mask_tmp=mask_tmp[indexes==j]
        mask_tmp=mask_tmp[:,tmp_index]
        df.loc["{}-{} m".format(start_depth[i],end_depth[i]),'Bias ({}, Coast)'.format(seasons_names[j-1])]="{:.2E}".format(np.mean(diff_tmp[mask_tmp]))
        df.loc["{}-{} m".format(start_depth[i],end_depth[i]),'Bias-UQ ({}, Coast)'.format(seasons_names[j-1])]="{:.2E}".format(np.mean(diff_uq_tmp[mask_tmp]))
        df.loc["{}-{} m".format(start_depth[i],end_depth[i]),'RMSD ({}, Coast)'.format(seasons_names[j-1])]="{:.2E}".format(np.mean(diff_tmp[mask_tmp]**2))
        df.loc["{}-{} m".format(start_depth[i],end_depth[i]),'RMSD-UQ ({}, Coast)'.format(seasons_names[j-1])]="{:.2E}".format(np.mean(diff_uq_tmp[mask_tmp]**2))
        print("Hello")
        del diff_tmp
        del mask_tmp
        del diff_uq_tmp
        diff_tmp=diff.copy()
        diff_tmp=diff_tmp[:,:,np.logical_not(coast_index)]
        diff_tmp=diff_tmp[indexes==j]
        diff_tmp=diff_tmp[:,tmp_index]
        diff_uq_tmp=diff_uq.copy()
        diff_uq_tmp=diff_uq_tmp[:,:,np.logical_not(coast_index)]
        diff_uq_tmp=diff_uq_tmp[indexes==j]
        diff_uq_tmp=diff_uq_tmp[:,tmp_index]
        mask_tmp=mask_intersected.copy()
        mask_tmp=mask_tmp[:,:,np.logical_not(coast_index)]
        mask_tmp=mask_tmp[indexes==j]
        mask_tmp=mask_tmp[:,tmp_index]
        df.loc["{}-{} m".format(start_depth[i],end_depth[i]),'Bias ({}, Opensea)'.format(seasons_names[j-1])]="{:.2E}".format(np.mean(diff_tmp[mask_tmp]))
        df.loc["{}-{} m".format(start_depth[i],end_depth[i]),'Bias-UQ ({}, Opensea)'.format(seasons_names[j-1])]="{:.2E}".format(np.mean(diff_uq_tmp[mask_tmp]))
        df.loc["{}-{} m".format(start_depth[i],end_depth[i]),'RMSD ({}, Opensea)'.format(seasons_names[j-1])]="{:.2E}".format(np.mean(diff_tmp[mask_tmp]**2))
        df.loc["{}-{} m".format(start_depth[i],end_depth[i]),'RMSD-UQ ({}, Opensea)'.format(seasons_names[j-1])]="{:.2E}".format(np.mean(diff_uq_tmp[mask_tmp]**2))
        del diff_tmp
        del mask_tmp
        del diff_uq_tmp


df.to_csv(config.plot_path+"/validation_summary.csv", index=True)  


mean_all=mean_all[:,0,:,:]
data_all=test_dataset[:,0,:,:]
diff=diff[:,0,:,:]
diff_uq=diff_uq[:,0,:,:]
mask=np.logical_not(mask_intersected[:,0,:,:])
mean_all[mask]=np.nan
data_all[mask]=np.nan

for i in trange(len(mean_all)):
    gs=GridSpec(4,4)
    ax00 = plt.subplot(gs[:2, :2])
    ax10 = plt.subplot(gs[2:, :2])
    ax01=  plt.subplot(gs[:2, 2:])
    ax11=  plt.subplot(gs[2:, 2:])

    max_val=max(np.nanmax(data_all[i]),np.nanmax(mean_all[i]))
    min_val=max(np.nanmin(data_all[i]),np.nanmin(mean_all[i]))
    plot_0=ax00.imshow(np.ma.masked_array(mean_all[i],mask[i]),origin="lower",cmap="jet",vmax=max_val,vmin=min_val)
    plt.colorbar(plot_0, ax=ax00)
    ax00.axes.xaxis.set_visible(False)
    ax00.axes.yaxis.set_visible(False)
    ax00.title.set_text("Mean Predictor")
    plot_1=ax01.imshow(np.ma.masked_array(data_all[i],mask[i]),origin="lower",cmap="jet",vmax=max_val,vmin=min_val)    
    plt.colorbar(plot_1, ax=ax01)
    ax01.axes.xaxis.set_visible(False)
    ax01.axes.yaxis.set_visible(False)
    ax01.title.set_text("True")
    plot_2=ax10.imshow(np.ma.masked_array(diff[i],mask[i]),origin="lower",cmap="jet")
    plt.colorbar(plot_2, ax=ax10)
    ax10.axes.xaxis.set_visible(False)
    ax10.axes.yaxis.set_visible(False)
    ax10.title.set_text("Difference")
    plot_3=ax11.imshow(np.ma.masked_array(diff_uq[i],mask[i]),origin="lower",cmap="jet")
    plt.colorbar(plot_3, ax=ax11)
    ax11.axes.xaxis.set_visible(False)
    ax11.axes.yaxis.set_visible(False)
    ax11.title.set_text("Difference from the CI")
    plt.suptitle(config.variable+"-"+config.basin+"-"+(datetime.strptime(config.start_date, "%Y-%m-%d")+timedelta(days=i+1)).strftime("%Y-%m-%d"))
    plt.savefig(config.plot_path+"/validation_plots/validation_{}.pdf".format(i+1))
    plt.clf()
