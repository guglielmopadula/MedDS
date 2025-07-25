
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

lb=np.load(config.download_path+"/lb.npy")
ub=np.load(config.download_path+"/ub.npy")
mask=np.logical_not(np.load(config.download_re_path+"/mask.npy"))
mask=np.repeat(np.expand_dims(mask,0),(len(mean)),0)


test_data=test_dataset[np.logical_not(mask)].reshape(config.target_prediction_time,-1)

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


seasons = ['Autumn', 'Summer','Winter','Spring']

# Create a DataFrame with random data (you can replace this with actual data)
data = {
    '0-10 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '10-30 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '30-60 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '60-100 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    '100-150 m': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
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




#Copernicus classification of seasons, Winter is from Jan to Apr and summer from Jun to Sept
seasons_names=["Winter","Spring","Summer","Autumn"]


seasons=np.array([1,1,1,
                  2,2,2,
                  3,3,3,
                  4,4,4])
indexes=np.zeros(config.target_prediction_time,dtype=np.int64)
date_obj = datetime.strptime(config.start_date, "%Y-%m-%d")

dates=np.zeros(config.target_prediction_time,dtype='U100')


for i in range(config.target_prediction_time):
    tmp = date_obj + timedelta(days=1+i)
    dates[i]=datetime.strftime(tmp,"%Y-%m-%d")
    indexes[i]=get_season(tmp)


start_depth=np.array([0,10,30,60,100])
end_depth=np.array([10,30,60,100,150])
max_depth=depth[np.sum(np.logical_not(mask[0]),axis=0)-1]
coast_index=max_depth<200



for j in range(1,5):
    mean_season=mean_all[indexes==j]
    re_season=re_all[indexes==j]
    if np.prod(mean_season.shape)>0:
        test_season=data_all[indexes==j]
        mask_season=mask_intersected[indexes==j]
        mean_rec_vec=np.zeros(len(mean_season))
        std_rec_vec=np.zeros(len(mean_season))
        mean_true_vec=np.zeros(len(mean_season))
        std_true_vec=np.zeros(len(mean_season))
        corr=np.zeros(len(mean_season))

        for k in range(len(mean_season)):
            tmp_mask=mask_season[k]
            tmp_mean=mean_season[k]
            tmp_test=test_season[k]
            mean_rec_vec[k]=np.mean(tmp_mean[tmp_mask])
            mean_true_vec[k]=np.mean(tmp_test[tmp_mask])
            std_rec_vec[k]=np.std(tmp_mean[tmp_mask])
            std_true_vec[k]=np.std(tmp_test[tmp_mask])
            cov_tmp=np.mean((tmp_mean[tmp_mask]-mean_rec_vec[k])*(tmp_test[tmp_mask]-mean_true_vec[k]))
            corr[k]=cov_tmp/(std_true_vec[k]*std_rec_vec[k])

        plt.plot(np.arange(len(mean_true_vec)),mean_true_vec,label="Copernicus Mean",color="red")
        plt.plot(np.arange(len(mean_rec_vec)),mean_rec_vec,label="DS Mean",color="blue")
        plt.fill_between(np.arange(len(mean_true_vec)),mean_true_vec-1.96*std_true_vec,mean_true_vec+1.96*std_true_vec, color="red", alpha=0.5)
        plt.fill_between(np.arange(len(mean_true_vec)),mean_rec_vec-1.96*std_rec_vec,mean_rec_vec+1.96*std_rec_vec, color="blue", alpha=0.5)
        plt.legend()
        xtick=np.arange(len(mean_true_vec))[::20]
        date_tmp=dates[indexes==j]

        plt.xticks(xtick,date_tmp[::20],rotation=90)
        plt.suptitle(config.variable+'  '+config.basin)
        plt.tight_layout()
        plt.savefig(config.plot_path+"/validation_plots/validation_season_{}.pdf".format(j))
        plt.clf()
        plt.plot(np.arange(len(mean_true_vec)),corr,label="corr")
        plt.plot(np.arange(len(mean_true_vec)),0.6*np.ones_like(np.arange(len(mean_true_vec))),'--',label="threshold")
        plt.legend()
        xtick=np.arange(len(mean_true_vec))[::20]
        date_tmp=dates[indexes==j]
        plt.xticks(xtick,date_tmp[::20],rotation=90)
        plt.suptitle(config.variable+'  '+config.basin)
        plt.tight_layout()
        plt.savefig(config.plot_path+"/validation_plots/corr_season_{}.pdf".format(j))
        plt.clf()


        mean_rec_vec=mean_season[mask_season]
        mean_true_vec=test_season[mask_season]
        mean_rec_vec=mean_rec_vec.reshape(mean_season.shape[0],-1)
        mean_true_vec=mean_true_vec.reshape(mean_season.shape[0],-1)
        cov_xy=np.mean((mean_rec_vec-np.mean(mean_rec_vec,axis=0))*(mean_true_vec-np.mean(mean_true_vec,axis=0)),axis=0)
        var_x=np.var(mean_rec_vec,axis=0)
        var_y=np.var(mean_true_vec,axis=0)
        coff=cov_xy/np.sqrt(var_x*var_y)
        coff=np.nan_to_num(coff)
        #arr=np.zeros_like(mask_season[0,0],dtype=np.float64)
        #arr[mask_season[0,0]]=coff
        #arr=np.ma.masked_array(arr,np.logical_not(mask_season[0,0]))
        arr=np.zeros_like(mask_season[0],dtype=np.float64)
        arr[mask_season[0]]=coff
        arr=np.ma.masked_array(arr[0],np.logical_not(mask_season[0,0]))
        plt.imshow(arr,origin="lower")
        plt.suptitle(config.variable+'  '+config.basin)
        plt.colorbar()
        plt.savefig(config.plot_path+"/validation_plots/spat_corr_season_{}.pdf".format(j))
        plt.clf()
        plt.imshow(arr>0.6,origin="lower")
        plt.suptitle(config.variable+'  '+config.basin)
        plt.colorbar()
        plt.savefig(config.plot_path+"/validation_plots/spat_corr_season_{}_bin.pdf".format(j))
        plt.clf()


        mean_rec_vec=np.zeros(len(mean_season))
        std_rec_vec=np.zeros(len(mean_season))
        mean_true_vec=np.zeros(len(mean_season))
        std_true_vec=np.zeros(len(mean_season))
        corr=np.zeros(len(mean_season))


        for k in range(len(mean_season)):
            tmp_mask=mask_season[k]
            tmp_mean=mean_season[k]
            tmp_test=test_season[k]
            tmp_re=re_season[k]
            mean_rec_vec[k]=np.mean(tmp_mean[tmp_mask]-tmp_re[tmp_mask])
            mean_true_vec[k]=np.mean(tmp_test[tmp_mask]-tmp_re[tmp_mask])
            std_rec_vec[k]=np.sqrt(np.mean((tmp_mean[tmp_mask]-tmp_re[tmp_mask])*(tmp_mean[tmp_mask]-tmp_re[tmp_mask])))
            std_true_vec[k]=np.sqrt(np.mean((tmp_test[tmp_mask]-tmp_re[tmp_mask])*(tmp_test[tmp_mask]-tmp_re[tmp_mask])))
            cov_tmp=np.mean((tmp_mean[tmp_mask]-tmp_re[tmp_mask])*(tmp_test[tmp_mask]-tmp_re[tmp_mask]))
            corr[k]=cov_tmp/(std_true_vec[k]*std_rec_vec[k])

        plt.plot(np.arange(len(mean_true_vec)),corr,label="corr")
        plt.plot(np.arange(len(mean_true_vec)),0.6*np.ones_like(np.arange(len(mean_true_vec))),'--',label="threshold")
        plt.legend()
        xtick=np.arange(len(mean_true_vec))[::20]
        date_tmp=dates[indexes==j]
        plt.xticks(xtick,date_tmp[::20],rotation=90)
        plt.suptitle(config.variable+'  '+config.basin)
        plt.tight_layout()
        plt.savefig(config.plot_path+"/validation_plots/acc_season_{}.pdf".format(j))
        plt.clf()




        mean_rec_vec=mean_season[mask_season]-re_season[mask_season]
        mean_true_vec=test_season[mask_season]-re_season[mask_season]
        mean_rec_vec=mean_rec_vec.reshape(mean_season.shape[0],-1)
        mean_true_vec=mean_true_vec.reshape(mean_season.shape[0],-1)
        cov_xy=np.mean((mean_rec_vec)*(mean_true_vec),axis=0)
        var_x=np.mean((mean_rec_vec)*(mean_rec_vec),axis=0)
        var_y=np.mean((mean_true_vec)*(mean_true_vec),axis=0)
        coff=cov_xy/np.sqrt(var_x*var_y)
        arr=np.zeros_like(mask_season[0],dtype=np.float64)
        arr[mask_season[0]]=coff
        #arr=np.zeros_like(mask_season[0,0],dtype=np.float64)
        #arr[mask_season[0,0]]=coff
        arr=np.ma.masked_array(arr[0],np.logical_not(mask_season[0,0]))
        plt.imshow(arr,origin="lower")
        plt.suptitle(config.variable+'  '+config.basin)
        plt.colorbar()
        plt.savefig(config.plot_path+"/validation_plots/spat_acc_season_{}.pdf".format(j))
        plt.clf()
        plt.imshow(arr>0.6,origin="lower")
        plt.suptitle(config.variable+'  '+config.basin)
        plt.colorbar()
        plt.savefig(config.plot_path+"/validation_plots/spat_acc_season_{}_bin.pdf".format(j))
        plt.clf()
