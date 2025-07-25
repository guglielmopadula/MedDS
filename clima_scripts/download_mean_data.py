from staticvariables import *
import os 
import numpy as np
import copernicusmarine
from datetime import datetime, timedelta
import sys
import xarray as xr

import importlib.util
def module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("miao", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

eps=1e-05


years=np.arange(1999,2023)


variable=sys.argv[1]
k=int(sys.argv[2])

i=years[k]

if not os.path.exists("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/mean_{}.nc".format(variable,i)):
    dataset=copernicusmarine.subset(
        dataset_id=reanalysis_names[variables_dict[variable][4]].format(variables_dict[variable][3]),
        variables=[variable],
        start_datetime="{}-01-01".format(i),
        end_datetime="{}-12-31".format(i),
        minimum_longitude=-5.55-eps,
        maximum_longitude=36.30-eps,
        minimum_latitude=30.17+eps,
        maximum_latitude=45.98+eps,
        output_filename="mean_{}.nc".format(i),
        output_directory="/g100_scratch/userexternal/gpadula0/ClimaScratch/{}".format(variable),
        chunk_size_limit=10
        )

if not os.path.exists("/g100_scratch/userexternal/gpadula0/ClimaScratch/{}/static.nc".format(variable)):
    if np.sum(k)<1:
        copernicusmarine.subset(
                dataset_id=static_dataset.format(variables_dict[variable][4]),
                output_filename = "static",
                output_directory ="/g100_scratch/userexternal/gpadula0/ClimaScratch/{}".format(variable)
,               start_datetime="{}-01-01".format(i),
                end_datetime="{}-12-31".format(i),
                minimum_longitude=-5.55-eps,
                maximum_longitude=36.30-eps,
                minimum_latitude=30.17+eps,
                maximum_latitude=45.98+eps, 
                force_download=True,
                chunk_size_limit=100
            )
