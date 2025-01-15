import numpy as np
from staticvariables import *
import sys
config=__import__(sys.argv[1])

if config.device=="cpu":
    import numpy as cp
elif config.device=="gpu":
    import cupy as cp

else:
    print("Unkwown device")
    assert False
from scipy.stats import truncnorm, binom


lb=variables_dict[config.variable][0]
ub=variables_dict[config.variable][1]

training_data=cp.array(np.load(config.download_path+"/training_data.npy"))
training_data_mean=cp.mean(training_data,axis=0)
training_data=training_data-training_data_mean

U,S,V=cp.linalg.svd(training_data,full_matrices=False)
S=S/cp.sqrt(len(training_data))
rank=cp.argmin(cp.abs(S-1/10*S[0]))+1
if config.device=="gpu":
    rank=rank.get()
S=S[:rank]
U=U[:,:rank]
V=V[:rank]
U=cp.sqrt(len(training_data))*U
W=cp.diag(S)@V
Winv=V.T@cp.diag(1/(S))

err=U@W-training_data
err_std=cp.std(err,axis=0)

U2=U**2

_,S2,_=cp.linalg.svd(U2.T@U2,full_matrices=False)

W2=cp.linalg.inv(U2.T@U2+S2[0]*cp.eye(U2.shape[1]))@U2.T@err


len_I=training_data.shape[0]-1


def k(x,y):
    return cp.exp(-((x.T.reshape(x.shape[1],x.shape[0],1)-y.T.reshape(y.shape[1],1,y.shape[0]))**2)/(2))+3*cp.exp(-((x.T.reshape(x.shape[1],x.shape[0],1)-y.T.reshape(y.shape[1],1,y.shape[0]))**2)/(2*(1e-08)))


U_train=U[:-config.timestep]
U_test=U[-config.timestep:]
k_train=k(U_train,U_train)
k_test_train=k(U_test,U_train)
k_test=k(U_test,U_test)

def batched_matmul(X,Y):
    return cp.einsum('Bik,Bkj ->Bij', X, Y)

#k_train=k(U,U)


b=cp.zeros((len(U_train),len(U)))

for i in range(len(U_train)):
    if i<config.timestep:
        b[i,i]=-1
        b[i,i+config.timestep]=1
    
    if i>=config.timestep and i+config.timestep<len(U):
        b[i,i-config.timestep]=-1/2
        b[i,i+config.timestep]=1/2
    

diffs_train=b@U



k_train_reg=k_train#unc*np.eye(len_I).reshape(1,len_I,len_I)#+reg*np.eye(len_I).reshape(1,len_I,len_I)
num_times=k_train_reg.shape[1]

#print(k_train.shape)

#diff=U[timestep:]-U[:-timestep]

matrix=cp.linalg.solve(k_train_reg,diffs_train.T.reshape(rank,num_times,1))

data_rec_test=cp.zeros((config.target_prediction_time,training_data.shape[1]))
std_rec_test=cp.zeros((config.target_prediction_time,training_data.shape[1]))
data_U_test=cp.zeros((config.target_prediction_time,U.shape[1]))
std_U_test=cp.zeros((config.target_prediction_time,U.shape[1]))
mode_rec_test=cp.zeros((config.target_prediction_time,training_data.shape[1]))
ub_rec_test=cp.zeros((config.target_prediction_time,training_data.shape[1]))
lb_rec_test=cp.zeros((config.target_prediction_time,training_data.shape[1]))

training_std=cp.std(training_data,axis=0)


def cdf(x,mu,sigma):
    mu_tmp=mu
    sigma_tmp=sigma
    alpha=(lb-mu_tmp)/(sigma_tmp+1e-06)
    beta=(ub-mu_tmp)/(sigma_tmp+1e-06)
    return truncnorm.cdf(x,loc=mu_tmp,a=alpha,b=beta,scale=(sigma_tmp+1e-06))




def ppf(x,mu,sigma):
    mu_tmp=mu
    sigma_tmp=sigma
    alpha=(lb-mu_tmp)/(sigma_tmp+1e-06)
    beta=(ub-mu_tmp)/(sigma_tmp+1e-06)
    return truncnorm.ppf(x,loc=mu_tmp,a=alpha,b=beta,scale=(sigma_tmp+1e-06))

def batched_matmul(X,Y):
    return cp.einsum('Bik,Bkj ->Bij', X, Y)

data_U_test[:config.timestep]=((batched_matmul(k_test_train,matrix)).reshape(rank,-1).T)+U_test
tmp_U_test=cp.sqrt(k_test-batched_matmul(k_test_train,cp.linalg.solve(k_train_reg,cp.transpose(k_test_train,(0,2,1)))))
std_U_test[:config.timestep]=cp.diagonal(tmp_U_test,0,1,2).T
num_steps=config.target_prediction_time//config.timestep
if num_steps>1:
    for i in range(1,num_steps):
        U_tmp=cp.concatenate((U_test[len(U_test)-(i)*config.timestep:],data_U_test[:i*config.timestep]))
        U_tmp=U_tmp[-2*config.timestep:]
        k_test_train=k(U_tmp,U[:-config.timestep])
        k_test_test=k(U_tmp,U_tmp)
        k_test_train_prev=k_test_train[:,:config.timestep]
        k_test_train_now=k_test_train[:,config.timestep:]
        k_test_test_prev=k_test_test[:,:config.timestep,:config.timestep]
        k_test_test_now=k_test_test[:,config.timestep:,config.timestep:]
        data_U_test[(i)*config.timestep:(i+1)*config.timestep]=data_U_test[(i-1)*config.timestep:(i)*config.timestep]+3/2*(batched_matmul(k_test_train_now,matrix)).reshape(rank,-1).T-1/2*(batched_matmul(k_test_train_prev,matrix)).reshape(rank,-1).T
        tmp_U_test=3/2*(k_test_test_now-batched_matmul(k_test_train_now,cp.linalg.solve(k_train_reg,cp.transpose(k_test_train_now,(0,2,1)))))#-1/2*(k_test_test_prev-batched_matmul(k_test_train_prev,cp.linalg.solve(k_train_reg,cp.transpose(k_test_train_prev,(0,2,1)))))
        tmp_U_test=cp.diagonal(tmp_U_test,0,1,2).T
        std_U_test[(i)*config.timestep:(i+1)*config.timestep]=cp.sqrt(tmp_U_test+std_U_test[(i-1)*config.timestep:(i)*config.timestep]**2)


lb_qf=binom.ppf(q=0.005,n=rank,p=0.95)/rank
ub_qf=binom.ppf(q=0.995,n=rank,p=0.95)/rank

data_rec_test=data_U_test@W+training_data_mean+(data_U_test**2)@W2
std_rec_test=cp.sqrt((std_U_test**2)@(W**2)+(4*data_U_test**2*std_U_test**2+2*std_U_test**4)@(W2**2))

if config.device=="gpu":
    data_rec_test=data_rec_test.get()
    std_rec_test=std_rec_test.get()
    data_U_test=data_U_test.get()
    std_U_test=std_U_test.get()



perc=0.95
mode_rec_test=ub*(data_rec_test>ub)+lb*(data_rec_test<lb)+(data_rec_test)*(data_rec_test<ub)*(data_rec_test>lb)
mode_cdf=cdf(mode_rec_test,data_rec_test,std_rec_test)
lb_mode_cdf=mode_cdf-perc/2
ub_mode_cdf=mode_cdf+perc/2

ub_mode_cdf[lb_mode_cdf<0]=ub_mode_cdf[lb_mode_cdf<0]-lb_mode_cdf[lb_mode_cdf<0]
lb_mode_cdf[lb_mode_cdf<0]=0
lb_mode_cdf[ub_mode_cdf>1]=lb_mode_cdf[ub_mode_cdf>1]-(1-ub_mode_cdf[ub_mode_cdf>1])
ub_mode_cdf[ub_mode_cdf>1]=1

ub_rec_test=ppf(ub_mode_cdf,data_rec_test,std_rec_test)
lb_rec_test=ppf(lb_mode_cdf,data_rec_test,std_rec_test)
np.save(config.download_path+"/mean.npy",data_rec_test)
np.save(config.download_path+"/lb.npy",lb_rec_test)
np.save(config.download_path+"/ub.npy",ub_rec_test)


#These are needed for the theoretical validation
np.save(config.download_path+"/latent_mean.npy",data_U_test) 
np.save(config.download_path+"/latent_std.npy",std_U_test)
np.save(config.download_path+"/matrix.npy",Winv)
