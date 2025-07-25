import matplotlib.pyplot as plt
import sys
from staticvariables import *
import numpy as np
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime, timedelta
import importlib.util
def module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("miao", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config=module_from_file(sys.argv[1])



mean=np.load(config.download_path+"/mean.npy")
lb=np.load(config.download_path+"/lb.npy")
ub=np.load(config.download_path+"/ub.npy")
mask=np.logical_not(np.load(config.download_re_path+"/mask.npy"))

if not os.path.exists(config.plot_path+"/prediction_plots"):
    os.makedirs(config.plot_path+"/prediction_plots")


mask=np.repeat(np.expand_dims(mask,0),(len(mean)),0)
mean_all=np.zeros(mask.shape)
lb_all=np.zeros(mask.shape)
ub_all=np.zeros(mask.shape)
mean_all[np.logical_not(mask)]=mean.reshape(-1)
lb_all[np.logical_not(mask)]=lb.reshape(-1)
ub_all[np.logical_not(mask)]=ub.reshape(-1)
mean_all[mask]=np.nan
lb_all[mask]=np.nan
ub_all[mask]=np.nan
variable=config.variable
name=variables_dict[variable][3]
basin=config.basin

if len(mask.shape)==4:
    lb_all=lb_all[:,0,:,:]
    ub_all=ub_all[:,0,:,:]
    mean_all=mean_all[:,0,:,:]
    mask=mask[:,0]

for i in range(len(mean_all)):
    gs=GridSpec(4,4)
    ax00 = plt.subplot(gs[:2, :2])
    ax10 = plt.subplot(gs[2:, :2])
    ax01=  plt.subplot(gs[1:3, 2:])
    max_val=max(np.nanmax(mean_all[i]),np.nanmax(ub_all[i]),np.nanmax(lb_all[i]))
    min_val=min(np.nanmin(mean_all[i]),np.nanmin(ub_all[i]),np.nanmin(lb_all[i]))
    plot_0=ax00.imshow(np.ma.masked_array(mean_all[i],mask[i]),origin="lower",cmap="jet",vmax=max_val,vmin=min_val)
    plt.colorbar(plot_0, ax=ax00)
    ax00.axes.xaxis.set_visible(False)
    ax00.axes.yaxis.set_visible(False)
    ax00.title.set_text("Mean Predictor")
    plot_1=ax01.imshow(np.ma.masked_array(ub_all[i],mask[i]),origin="lower",cmap="jet",vmax=max_val,vmin=min_val)    
    plt.colorbar(plot_1, ax=ax01)
    ax01.axes.xaxis.set_visible(False)
    ax01.axes.yaxis.set_visible(False)
    ax01.title.set_text("Upper Bound")
    plot_2=ax10.imshow(np.ma.masked_array(lb_all[i],mask[i]),origin="lower",cmap="jet",vmax=max_val,vmin=min_val)
    plt.colorbar(plot_2, ax=ax10)
    ax10.axes.xaxis.set_visible(False)
    ax10.axes.yaxis.set_visible(False)
    ax10.title.set_text("Lower Bound")
    plt.suptitle(variable+"-"+config.basin+"-"+(datetime.strptime(config.start_date, "%Y-%m-%d")+timedelta(days=i+1)).strftime("%Y-%m-%d"))
    plt.savefig(config.plot_path+"/prediction_plots/prediction_{}.pdf".format(i+1))
    plt.clf()

