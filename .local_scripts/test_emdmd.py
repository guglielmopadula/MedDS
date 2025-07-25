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
from pydmd import EDMD
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

"""
vec_train_all=np.zeros((vec_train.shape[0],*mask.shape))
vec_re_train_all=np.zeros((vec_train.shape[0],*mask.shape))
vec_re_test_all=np.zeros((testing_re_mean.shape[0],*mask.shape))

vec_train_all[:,np.logical_not(mask)]=vec_train
vec_re_train_all[:,np.logical_not(mask)]=training_re_mean
vec_re_test_all[:,np.logical_not(mask)]=testing_re_mean

vec_train=vec_train_all[:,0,np.logical_not(mask[0])]
training_re_mean=vec_re_train_all[:,0,np.logical_not(mask[0])]
testing_re_mean=vec_re_test_all[:,0,np.logical_not(mask[0])]


test_output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)

l1=mask.shape[1]
l2=mask.shape[2]

test_dataset=nc.Dataset(config.download_path+"/"+test_output_filename+".nc")
test_dataset=test_dataset.variables[config.variable][:]
vec_test=test_dataset.data[:,0,np.logical_not(mask[0])].reshape(config.target_prediction_time,-1)
"""
test_output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)

l1=mask.shape[1]
l2=mask.shape[2]

test_dataset=nc.Dataset(config.download_path+"/"+test_output_filename+".nc")
test_dataset=test_dataset.variables[config.variable][:]
vec_test=test_dataset.data[:,np.logical_not(mask)].reshape(config.target_prediction_time,-1)





start=time()
dmd=EDMD(svd_rank=20,kernel_metric="rbf")

dmd.fit(vec_train.T)
dmd.dmd_time["tend"]=config.n_train+config.target_prediction_time
data_rec_test=dmd.reconstructed_data
data_rec_test=data_rec_test.T
data_rec_test=data_rec_test[config.n_train+1:]
end=time()
print(np.mean(np.abs(vec_test-data_rec_test))/(np.max(vec_test)-np.min(vec_test)))
print(np.mean(np.sqrt(np.mean(np.abs((vec_test-data_rec_test))**2,axis=1)))/(np.max(vec_test)-np.min(vec_test)))
print(end-start)


