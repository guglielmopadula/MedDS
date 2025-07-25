
from staticvariables import *
import os 
import numpy as np
import subprocess
import copernicusmarine
from datetime import datetime, timedelta
import sys
import shutil
import netCDF4 as nc
import xarray as xr
from tqdm import trange
import importlib.util


variable=sys.argv[1]

if not os.path.exists("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mean_all.nc".format(variable)):
    if not os.path.exists("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mean_{}.nc".format(variable,"1999")):

        print("Download data first")
    else:
        dataset=nc.Dataset("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mean_{}.nc".format(variable,"1999")).variables[variable]
        mask=dataset[:].mask
        mask=np.logical_not(mask[0])
        mask_analysis=nc.Dataset("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/static.nc".format(variable)).variables["mask"][:]
        if len(mask.shape)==2:
            mask_analysis=mask_analysis[0]
        mask_analysis=np.array(mask_analysis,dtype=bool)
        mask=mask*mask_analysis
        mask=np.logical_not(mask)
        np.save("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mask.npy".format(variable),mask)
        data=dataset[:].data
        data=data[:,np.logical_not(mask)]
        mean=np.zeros((366,data.shape[1]))
        counter=np.zeros(366)
        for i in trange(1999,2023):
            dataset=nc.Dataset("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mean_{}.nc".format(variable,i)).variables[variable]
            data=dataset[:].data
            data=data[:,np.logical_not(mask)]
            tmp=np.ones((len(data),))
            if len(tmp)<366:
                tmp=np.insert(tmp,59,0,0)
            if len(data)<366:
                data=np.insert(data,59,0,0)                
            counter=counter+tmp
            mean=mean+data
        mean=mean/counter.reshape(-1,1)
        subprocess.run(["cp", "/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mean_2020.nc".format(variable), "/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mean_all.nc".format(variable)])
        dataset_fin=nc.Dataset("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mean_all.nc".format(variable),mode="r+")
        dataset_fin.variables[variable][:].data[:,np.logical_not(mask)]=mean
        dataset_fin.close()
