import numpy as np
from scipy.stats import truncnorm, binom
import netCDF4 as nc
import sys
config=__import__(sys.argv[1])

test_output_filename="test_{}_{}_{}_{}".format(config.variable,config.basin,config.start_date,config.target_prediction_time)
test_dataset=nc.Dataset(config.download_path+"/"+test_output_filename+".nc")
training_data_mean=np.load(config.download_path+"/train_mean.npy")
test_mean=np.load(config.download_path+"/re_testing_mean.npy")
mask=np.load(config.download_re_path+"/mask.npy")
data_rec_test=np.load(config.download_path+"/mean.npy")

test_dataset=test_dataset.variables[config.variable][:]
testing_data=test_dataset.data[:,np.logical_not(mask)].reshape(config.target_prediction_time,-1)

Winv=np.load(config.download_path+"/winv.npy")
Beta=np.load(config.download_path+"/beta.npy")
matrix=np.load(config.download_path+"/matrix.npy")
U=np.load(config.download_path+"/u.npy")
U_train=U[:-1]

def k(x,y):
    return 0.1*np.exp(-np.mean(np.abs(x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1)),axis=2)/(2))+0.9*np.exp(-np.mean(np.abs(x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1)),axis=2)/(2*(1e-08)))

k_train=k(U_train,U_train)

def mu(x):
    return k(x,U_train)@matrix

def sigma2(x):
    return k(x,x)-k(x,U_train)@np.linalg.solve(k_train,(k(U_train,x)))



U_test=np.concatenate((U[-1].reshape(1,-1),(testing_data-training_data_mean-test_mean)@Winv))

diff_U_test=U_test[1:]-U_test[:-1]

data_U_test=mu(U_test)

std_U_test=np.sqrt(sigma2(U_test))
std_U_test=np.diagonal(np.sqrt(sigma2(U_test)),axis1=0,axis2=1)

data_U_test=mu(U_test)[:,:]
data_U_test=data_U_test[:-1]
std_U_test=std_U_test[:-1].reshape(-1,1)

flag=np.mean(np.logical_and(diff_U_test<data_U_test+Beta.reshape(1,-1)*std_U_test,diff_U_test>data_U_test-Beta.reshape(1,-1)*std_U_test))

if flag==1:
    print("Model is validated")

else:
    print("Model is not validated")

'''
print(data_U_test)
print(U_test)
rank=U_test.shape[1]
lb_qf=binom.ppf(q=0.005,n=rank,p=0.95)/rank
ub_qf=binom.ppf(q=0.995,n=rank,p=0.95)/rank


value=np.mean(np.logical_and(U_test<data_U_test+1.96*std_U_test,U_test>data_U_test-1.96*std_U_test))
if lb_qf<=value<=ub_qf:
    print("Model is validated")

else:
    print(lb_qf,value,ub_qf)
    print("Model is not validated")
'''
