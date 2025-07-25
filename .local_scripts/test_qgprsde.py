import numpy as np
from staticvariables import *
import sys
from time import time
from scipy.stats import truncnorm, binom, rv_histogram
from scipy.special import roots_hermite
from sklearn.utils.extmath import randomized_svd
from numpy import histogram
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import netCDF4 as nc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qgprsde import predict


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

'''
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

'''

test_output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)

l1=mask.shape[1]
l2=mask.shape[2]

test_dataset=nc.Dataset(config.download_path+"/"+test_output_filename+".nc")
test_dataset=test_dataset.variables[config.variable][:]
vec_test=test_dataset.data[:,np.logical_not(mask)].reshape(config.target_prediction_time,-1)


rank=20
rank2=2
start=time()
data_rec_test,var_rec_test, err_train, var_train=predict(vec_train-training_re_mean,config.target_prediction_time,rank,rank2)
data_rec_test=data_rec_test+testing_re_mean
end=time()


data_rec_test=data_rec_test.reshape(config.target_prediction_time,-1)

print(np.mean(np.abs(vec_test-data_rec_test))/(np.max(data_rec_test)-np.min(data_rec_test)))
print(np.linalg.norm(vec_test-data_rec_test)/(np.linalg.norm(vec_test)))
print(end-start)


print(np.mean(vec_test<data_rec_test+4.47*np.sqrt(var_rec_test)))
print(np.mean(vec_test>data_rec_test-4.47*np.sqrt(var_rec_test)))




fig,ax=plt.subplots(1,3)


tmp0=np.mean(data_rec_test,axis=0)
tmp1=np.mean(vec_test,axis=0)
tmp2=np.mean(np.abs(vec_test-data_rec_test))

max_val=np.maximum(np.max(tmp0),np.max(tmp1))
min_val=np.minimum(np.min(tmp0),np.min(tmp1))

'''
tmp0_all=np.zeros((l1,l2))
tmp0_all[np.logical_not(mask[0])]=tmp0

tmp1_all=np.zeros((l1,l2))
tmp1_all[np.logical_not(mask[0])]=tmp1

tmp2_all=np.zeros((l1,l2))
tmp2_all[np.logical_not(mask[0])]=tmp2

tmp0=np.ma.array(tmp0_all,mask=mask[0])
tmp1=np.ma.array(tmp1_all,mask=mask[0])
tmp2=np.ma.array(tmp2_all,mask=mask[0])



im0=ax[0].imshow(tmp0,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[0].set_title("Reconstructed")
im1=ax[1].imshow(tmp1,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[1].set_title("True")
im2=ax[2].imshow(tmp2,interpolation="gaussian")
ax[2].set_title("L1 diff")
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes('right', size='5%', pad=0.05)
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax0, orientation='vertical')
fig.colorbar(im1, cax=cax1, orientation='vertical')
fig.colorbar(im2, cax=cax2, orientation='vertical')
fig.tight_layout()
fig.savefig("qrgprsdeplot/test.pdf")
plt.clf()
'''


tmp0_all=np.zeros(mask.shape)
tmp0_all[np.logical_not(mask)]=tmp0

tmp1_all=np.zeros(mask.shape)
tmp1_all[np.logical_not(mask)]=tmp1

tmp2_all=np.zeros(mask.shape)
tmp2_all[np.logical_not(mask)]=tmp2

tmp0=np.ma.array(tmp0_all[0],mask=mask[0])
tmp1=np.ma.array(tmp1_all[0],mask=mask[0])
tmp2=np.ma.array(tmp2_all[0],mask=mask[0])



im0=ax[0].imshow(tmp0,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[0].set_title("Reconstructed")
im1=ax[1].imshow(tmp1,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[1].set_title("True")
im2=ax[2].imshow(tmp2,interpolation="gaussian")
ax[2].set_title("L1 diff")
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes('right', size='5%', pad=0.05)
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes('right', size='5%', pad=0.05)
divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax0, orientation='vertical')
fig.colorbar(im1, cax=cax1, orientation='vertical')
fig.colorbar(im2, cax=cax2, orientation='vertical')
fig.tight_layout()
fig.savefig("qrgprsdeplot/test.pdf",bbox_inches='tight')
plt.clf()




fig,ax=plt.subplots(1,3)

im0=ax[0].plot(np.mean(data_rec_test,axis=1).reshape(-1))
ax[0].set_title("Reconstructed")
im1=ax[1].plot(np.mean(vec_test,axis=1).reshape(-1))
ax[1].set_title("True")
im2=ax[2].plot(np.mean((np.abs(vec_test-data_rec_test)),axis=1).reshape(-1))
ax[2].set_title("L1 diff")
fig.tight_layout()
fig.savefig("qrgprsdeplot/test_time.pdf",bbox_inches='tight')
plt.clf()





tmp0=data_rec_test[-1]
tmp1=vec_test[-1]
tmp2=data_rec_test[-1]-4.47*np.sqrt(var_rec_test[-1])
tmp3=data_rec_test[-1]+4.47*np.sqrt(var_rec_test[-1])

tmp2=lb*(tmp2<lb)+tmp2*(tmp2>lb)

max_val=np.maximum(np.max(tmp0),np.max(tmp1))
min_val=np.minimum(np.min(tmp0),np.min(tmp1))

'''
tmp0_all=np.zeros((l1,l2))
tmp0_all[np.logical_not(mask[0])]=tmp0

tmp1_all=np.zeros((l1,l2))
tmp1_all[np.logical_not(mask[0])]=tmp1

tmp2_all=np.zeros((l1,l2))
tmp2_all[np.logical_not(mask[0])]=tmp2

tmp3_all=np.zeros((l1,l2))
tmp3_all[np.logical_not(mask[0])]=tmp3


tmp0=np.ma.array(tmp0_all,mask=mask[0])
tmp1=np.ma.array(tmp1_all,mask=mask[0])
tmp2=np.ma.array(tmp2_all,mask=mask[0])
tmp3=np.ma.array(tmp3_all,mask=mask[0])



fig,ax=plt.subplots(2,2)
im00=ax[0,0].imshow(tmp0,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[0,0].set_title("Reconstructed")
divider00 = make_axes_locatable(ax[0,0])
cax00 = divider00.append_axes('right', size='5%', pad=0.05)
im01=ax[0,1].imshow(tmp1,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[0,1].set_title("True")
divider01 = make_axes_locatable(ax[0,1])
cax01 = divider01.append_axes('right', size='5%', pad=0.05)
im10=ax[1,0].imshow(tmp2,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[1,0].set_title("Lower bound")
divider10 = make_axes_locatable(ax[1,0])
cax10 = divider10.append_axes('right', size='5%', pad=0.05)
im11=ax[1,1].imshow(tmp3,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[1,1].set_title("Upper bound")
divider11 = make_axes_locatable(ax[1,1])
cax11 = divider11.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im00, cax=cax00, orientation='vertical')
fig.colorbar(im01, cax=cax01, orientation='vertical')
fig.colorbar(im10, cax=cax10, orientation='vertical')
fig.colorbar(im11, cax=cax11, orientation='vertical')
fig.tight_layout()
fig.savefig("qrgprsdeplot/test_var.pdf")
'''

tmp0_all=np.zeros(mask.shape)
tmp0_all[np.logical_not(mask)]=tmp0




tmp1_all=np.zeros(mask.shape)
tmp1_all[np.logical_not(mask)]=tmp1



tmp2_all=np.zeros(mask.shape)
tmp2_all[np.logical_not(mask)]=tmp2

tmp3_all=np.zeros(mask.shape)
tmp3_all[np.logical_not(mask)]=tmp3


tmp0=np.ma.array(tmp0_all[0],mask=mask[0])
tmp1=np.ma.array(tmp1_all[0],mask=mask[0])
tmp2=np.ma.array(tmp2_all[0],mask=mask[0])
tmp3=np.ma.array(tmp3_all[0],mask=mask[0])

max_val=np.maximum(np.max(tmp0),np.max(tmp1))
min_val=np.minimum(np.min(tmp0),np.min(tmp1))


fig,ax=plt.subplots(2,2)
im00=ax[0,0].imshow(tmp0,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[0,0].set_title("Reconstructed")
divider00 = make_axes_locatable(ax[0,0])
cax00 = divider00.append_axes('right', size='5%', pad=0.05)
im01=ax[0,1].imshow(tmp1,vmax=max_val,vmin=min_val,interpolation="gaussian")
ax[0,1].set_title("True")
divider01 = make_axes_locatable(ax[0,1])
cax01 = divider01.append_axes('right', size='5%', pad=0.05)
im10=ax[1,0].imshow(tmp2,interpolation="gaussian")
ax[1,0].set_title("Lower bound")
divider10 = make_axes_locatable(ax[1,0])
cax10 = divider10.append_axes('right', size='5%', pad=0.05)
im11=ax[1,1].imshow(tmp3,interpolation="gaussian")
ax[1,1].set_title("Upper bound")
divider11 = make_axes_locatable(ax[1,1])
cax11 = divider11.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im00, cax=cax00, orientation='vertical')
fig.colorbar(im01, cax=cax01, orientation='vertical')
fig.colorbar(im10, cax=cax10, orientation='vertical')
fig.colorbar(im11, cax=cax11, orientation='vertical')
fig.tight_layout()
fig.savefig("qrgprsdeplot/test_var.pdf")
