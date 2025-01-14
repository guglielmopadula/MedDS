
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


config=__import__(sys.argv[1])

if not os.path.exists(config.download_path+"/validation_plots"):
    os.makedirs(config.download_path+"/validation_plots")


test_output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)
test_dataset=nc.Dataset(config.download_path+"/"+test_output_filename+".nc")
depth=nc.Dataset(config.download_path+"/"+"static.nc").variables["depth"][:].data
test_dataset=test_dataset.variables[config.variable][:]
pred_mask=test_dataset.mask
test_dataset=test_dataset.data
mean=np.load(config.download_path+"/mean.npy")
lb=np.load(config.download_path+"/lb.npy")
ub=np.load(config.download_path+"/ub.npy")
mask=np.load(config.download_path+"/mask.npy")
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
    '0-10 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '10-30 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '30-60 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '60-100 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '100-150 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
}

# Create the DataFrame
df = pd.DataFrame(data, index=seasons)

# Transpose the DataFrame
df_transposed = df.T

# Create the DataFrame
df = pd.DataFrame(data, index=seasons)
df=df.T

#Copernicus classification of seasons, Winter is from Jan to Apr and summer from Jun to Sept
seasons_names=["Winter","Spring","Summer","Autumn"]

seasons=np.array([1,1,1,1,2,3,3,3,3,4,4,4])
indexes=np.zeros(config.target_prediction_time,dtype=np.int64)
date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")
for i in range(config.target_prediction_time):
    tmp = date_obj + timedelta(days=1+i)
    indexes[i]=seasons[tmp.month-1]

start_depth=np.array([0,10,30,60,100])
end_depth=np.array([10,30,60,100,150])
max_depth=depth[np.sum(np.logical_not(mask[0]),axis=0)-1]
coast_index=max_depth<200

for i in trange(len(start_depth)):
    for j in range(1,5):
        tmp_index=np.logical_and(depth>start_depth[i],depth<end_depth[i])
        diff_tmp=diff.copy()
        diff_uq_tmp=diff_uq.copy()
        mask_tmp=mask_intersected.copy()
        diff_tmp=diff_tmp[:,:,coast_index]
        diff_tmp=diff_tmp[indexes==j]
        diff_tmp=diff_tmp[:,tmp_index]
        mask_tmp=mask_tmp[:,:,coast_index]
        mask_tmp=mask_tmp[indexes==j]
        mask_tmp=mask_tmp[:,tmp_index]
        diff_uq_tmp=diff_uq_tmp[:,:,coast_index]
        diff_uq_tmp=diff_uq_tmp[indexes==j]
        diff_uq_tmp=diff_uq_tmp[:,tmp_index]
        df['Bias ({}, Coast)'.format(seasons_names[j-1])]["{}-{} m".format(start_depth[i],end_depth[i])]="{:.2E}".format(np.mean(diff_tmp[mask_tmp]))
        df['Bias-UQ ({}, Coast)'.format(seasons_names[j-1])]["{}-{} m".format(start_depth[i],end_depth[i])]="{:.2E}".format(np.mean(diff_uq_tmp[mask_tmp]))
        df['RMSD ({}, Coast)'.format(seasons_names[j-1])]["{}-{} m".format(start_depth[i],end_depth[i])]="{:.2E}".format(np.mean(diff_tmp[mask_tmp]**2))
        df['RMSD-UQ ({}, Coast)'.format(seasons_names[j-1])]["{}-{} m".format(start_depth[i],end_depth[i])]="{:.2E}".format(np.mean(diff_uq_tmp[mask_tmp]**2))
        diff_tmp=diff.copy()
        diff_uq_tmp=diff_uq.copy()
        mask_tmp=mask_intersected.copy()
        diff_tmp=diff_tmp[:,:,np.logical_not(coast_index)]
        diff_tmp=diff_tmp[indexes==j]
        diff_tmp=diff_tmp[:,tmp_index]
        mask_tmp=mask_tmp[:,:,np.logical_not(coast_index)]
        mask_tmp=mask_tmp[indexes==j]
        mask_tmp=mask_tmp[:,tmp_index]
        diff_uq_tmp=diff_uq_tmp[:,:,np.logical_not(coast_index)]
        diff_uq_tmp=diff_uq_tmp[indexes==j]
        diff_uq_tmp=diff_uq_tmp[:,tmp_index]
        df['Bias ({}, Opensea)'.format(seasons_names[j-1])]["{}-{} m".format(start_depth[i],end_depth[i])]="{:.2E}".format(np.mean(diff_tmp[mask_tmp]))
        df['Bias-UQ ({}, Opensea)'.format(seasons_names[j-1])]["{}-{} m".format(start_depth[i],end_depth[i])]="{:.2E}".format(np.mean(diff_uq_tmp[mask_tmp]))
        df['RMSD ({}, Opensea)'.format(seasons_names[j-1])]["{}-{} m".format(start_depth[i],end_depth[i])]="{:.2E}".format(np.mean(diff_tmp[mask_tmp]**2))
        df['RMSD-UQ ({}, Opensea)'.format(seasons_names[j-1])]["{}-{} m".format(start_depth[i],end_depth[i])]="{:.2E}".format(np.mean(diff_uq_tmp[mask_tmp]**2))


df.to_csv(config.download_path+"/validation_summary.csv", index=True)  

mean_all=mean_all[:,0,:,:]
data_all=test_dataset[:,0,:,:]
diff=diff[:,0,:,:]
diff_uq=diff_uq[:,0,:,:]
mask=np.logical_not(mask_intersected[:,0,:,:])

for i in trange(len(mean_all)):
    gs=GridSpec(4,4)
    ax00 = plt.subplot(gs[:2, :2])
    ax10 = plt.subplot(gs[2:, :2])
    ax01=  plt.subplot(gs[:2, 2:])
    ax11=  plt.subplot(gs[2:, 2:])
    plot_0=ax00.imshow(np.ma.masked_array(mean_all[i],mask[i]),origin="lower",cmap="jet")
    plt.colorbar(plot_0, ax=ax00)
    ax00.axes.xaxis.set_visible(False)
    ax00.axes.yaxis.set_visible(False)
    ax00.title.set_text("Mean Predictor")
    plot_1=ax01.imshow(np.ma.masked_array(data_all[i],mask[i]),origin="lower",cmap="jet")    
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
    plt.savefig(config.download_path+"/validation_plots/validation_{}.pdf".format(i+1))
    plt.clf()