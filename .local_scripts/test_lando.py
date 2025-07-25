import numpy as np
from staticvariables import *
import sys
from time import time
from scipy.stats import truncnorm, binom, rv_histogram
from scipy.special import roots_hermite
from sklearn.utils.extmath import randomized_svd
from numpy import histogram
from scipy.integrate import solve_ivp
import netCDF4 as nc
from pydmd import LANDO
import importlib.util




def module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("miao", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config=module_from_file(sys.argv[1])


lb=variables_dict[config.variable][0]
ub=variables_dict[config.variable][1]
training_re_mean=np.load(config.download_path+"/re_training_mean.npy")
testing_re_mean=np.array(np.load(config.download_path+"/re_testing_mean.npy"))
mask=np.load(config.download_re_path+"/mask.npy")

vec_train=np.load(config.download_path+"/training_data.npy")

test_output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)

l1=mask.shape[1]
l2=mask.shape[2]

test_dataset=nc.Dataset(config.download_path+"/"+test_output_filename+".nc")
test_dataset=test_dataset.variables[config.variable][:]
vec_test=test_dataset.data[:,np.logical_not(mask)].reshape(config.target_prediction_time,-1)




start=time()
dmd=LANDO(svd_rank=20,kernel_metric="rbf")
t=np.arange(config.n_train+config.target_prediction_time)
t_train=t[:config.n_train]
t_test=t[config.n_train:]

dmd.fit(vec_train.T[:,:-1],vec_train.T[:,1:])

data_rec_test = dmd.predict(
    x0=vec_train[-1],
    tend=len(t_test),
    continuous=False,
    dt=1,
).T
end=time()

print(np.mean(np.abs(vec_test-data_rec_test))/(np.max(vec_test)-np.min(vec_test)))
print(np.mean(np.sqrt(np.mean(np.abs((vec_test-data_rec_test))**2,axis=1)))/(np.max(vec_test)-np.min(vec_test)))
print(end-start)

