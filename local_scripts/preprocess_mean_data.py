
from staticvariables import *
import os 
import numpy as np
import copernicusmarine
from datetime import datetime, timedelta
import sys
import netCDF4 as nc
import xarray as xr
import importlib.util


def module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("miao", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config=module_from_file(sys.argv[1])






if not os.path.exists(config.download_re_path+"/"+"reanalysis_mean.npy"):
    if not os.path.exists(config.download_re_path_all+"/"+"mean_all.nc"):
        print("Something went wrong")
    else:
        dataset=nc.Dataset(config.download_re_path_all+"/"+"mean_all.nc")
        lats = dataset.variables['latitude'][:]
        lons = dataset.variables['longitude'][:]
        latli = np.argmin( np.abs( lats - basins_dict[config.basin][2] ) )
        latui = np.argmin( np.abs( lats - basins_dict[config.basin][3] ) )
        lonli = np.argmin( np.abs( lons - basins_dict[config.basin][0] ) )
        lonui = np.argmin( np.abs( lons - basins_dict[config.basin][1] ) ) 
        dataset=dataset.variables[config.variable][:,:,latli:latui+1,lonli:lonui+1]
        mask=dataset[:].mask
        mask=np.logical_not(mask[0])
        '''
        mask=dataset[:].mask
        mask=np.logical_not(mask[0])
        mask_analysis=nc.Dataset(config.download_re_path+"/"+"static.nc").variables["mask"][:]
        if len(mask.shape)==2:
            mask_analysis=mask_analysis[0]
        mask_analysis=np.array(mask_analysis,dtype=bool)
        mask=mask*mask_analysis
        mask=np.logical_not(mask)
        np.save(config.download_re_path+"/"+"mask.npy",mask)
        '''
        data=dataset[:].data
        data=data[:,mask]
        '''
        mean=np.zeros((366,data.shape[1]))
        max_arr=-np.inf*np.ones((366,data.shape[1]))
        min_arr=np.inf*np.ones((366,data.shape[1]))
        counter=np.zeros(366)

        for i in range(1999,2023):
            dataset=nc.Dataset(config.download_re_path+"/"+"mean_{}.nc".format(i)).variables[config.variable]
            data=dataset[:].data
            data=data[:,np.logical_not(mask)]
            tmp=np.ones((len(data),))
            if len(tmp)<366:
                tmp=np.insert(tmp,59,0,0)
                print(tmp)
            if len(data)<366:
                data=np.insert(data,59,0,0)
                
            counter=counter+tmp
            mean=mean+data
        mean=mean/counter.reshape(-1,1)
        '''
        np.save(config.download_re_path+"/"+"mask.npy",mask)
        np.save(config.download_re_path+"/"+"reanalysis_mean.npy",data)


