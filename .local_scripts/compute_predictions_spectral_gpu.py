import numpy as np
from staticvariables import *
import sys
from time import time
config=__import__(sys.argv[1])
import cupy as cp
from cupyx.scipy import spatial

import skdim




start_time=time()
lb=variables_dict[config.variable][0]
ub=variables_dict[config.variable][1]

training_re_mean=(np.load(config.download_path+"/re_training_mean.npy"))



rank=4

training_data=np.load(config.download_path+"/training_data.npy")

training_data=cp.array(training_data)
training_re_mean=cp.array(training_re_mean)

def compute_cov(C):
    a = cp.arange(rank)
    b = cp.arange(rank)

    # Create the NxN matrix where A[i, j, 0] = a[i] and A[i, j, 1] = b[j]
    A = cp.zeros((len(a), len(b), 2))

    # Populate A[i, j, 0] = a[i] and A[i, j, 1] = b[j]
    A[:, :, 0] = a[:, cp.newaxis]  # broadcasting a[i]
    A[:, :, 1] = b[cp.newaxis, :]  # broadcasting b[j]


    A=A.reshape(-1,2)

    B = cp.zeros((len(A), len(A), 4),dtype=cp.int64)
    B[:,:,:2]=A[:,cp.newaxis]
    B[:,:,2:]=A[cp.newaxis:]
    B=B.reshape(rank,rank,rank,rank,-1)
    B0=B[:,:,:,:,0]
    B1=B[:,:,:,:,1]
    B2=B[:,:,:,:,2]
    B3=B[:,:,:,:,3]
    S=C[B0,B1]*C[B2,B3]+C[B0,B2]*C[B1,B3]+C[B0,B3]*C[B1,B2]
    return S.reshape(rank**2,rank**2)



'''
training_re_mean=training_re_mean/cp.sqrt(training_re_mean.shape[1])




Dist=(np.sum(training_re_mean**2,axis=1)).reshape(-1,1)+np.sum(training_re_mean**2,axis=1).reshape(1,-1)-2*cp.dot(training_re_mean,training_re_mean.T)
Dist=np.sqrt(Dist)
Dist=cp.nan_to_num(Dist)




Dist_2=Dist.copy()


k=1

#print(cp.sort(Dist_2,axis=1)[:,k])

eps=(cp.sum(((cp.sort(Dist_2,axis=1))[:,k]**4)/k))

r=eps**(1/4)

training_re_mean=training_re_mean/r

Dist=(np.sum(training_re_mean**2,axis=1)).reshape(-1,1)+np.sum(training_re_mean**2,axis=1).reshape(1,-1)-2*cp.dot(training_re_mean,training_re_mean.T)


D=cp.diag(cp.sum(Dist,axis=0))

L=D-Dist

M=np.diag(1/cp.sum(Dist,axis=0))@L
w,v=cp.linalg.eigh(M)

U_compl=v[:,1:5]
'''


training_data_res=training_data-training_re_mean


training_data_res=training_data_res/cp.sqrt(training_data_res.shape[1])


#training_data_res=training_data_res/(258405262319955)**1/4

Dist=(np.sum(training_data_res**2,axis=1)).reshape(-1,1)+np.sum(training_data_res**2,axis=1).reshape(1,-1)-2*cp.dot(training_data_res,training_data_res.T)
Dist=np.sqrt(Dist)
Dist=cp.nan_to_num(Dist)





Dist_2=Dist.copy()


k=1

#print(cp.sort(Dist_2,axis=1)[:,k])

eps=(cp.sum(((cp.sort(Dist_2,axis=1))[:,k]**4)/k))

r=eps**(1/4)

training_data_res=training_data_res/r

Dist=(np.sum(training_data_res**2,axis=1)).reshape(-1,1)+np.sum(training_data_res**2,axis=1).reshape(1,-1)-2*cp.dot(training_data_res,training_data_res.T)

def k_heat(r):
    return  1/(cp.sqrt(4*cp.pi*t)**(rank/2))*cp.exp(-r/(4*t))


t_exp=rank/2+4

t=(cp.log(training_data.shape[0])/training_data.shape[0])**(1/t_exp)
print(t)

Dist=k_heat(Dist)


print(np.linalg.cond(Dist.get()))

D=cp.diag(cp.sum(Dist,axis=0))

L=D-Dist

M=np.diag(1/cp.sum(Dist,axis=0))@L
w,v=cp.linalg.eigh(M)

U=v[:,1:5]

U=U*cp.sqrt(U.shape[0])

Up=U[1:]-U[:-1]



tmp=cp.sum((U.reshape(-1,1,4)-U.reshape(1,-1,4))**2,axis=2)

tmp_2=tmp.copy()
cp.fill_diagonal(tmp_2,999)
print(cp.sort(tmp_2,axis=1)[:,0])


lengthscale=cp.mean((cp.sort(tmp_2,axis=1))[:,0])



lengthscale=cp.minimum(cp.array(1.),lengthscale+1e-08)/2





cp.fill_diagonal(tmp,999)

index=0


while index<len(tmp):
    if cp.sum(tmp[index]<0.1*lengthscale)>0:
        U=cp.delete(U,index,axis=0)
        Up=cp.delete(Up,index,axis=0)
        tmp=cp.delete(tmp,index,0)
        tmp=cp.delete(tmp,index,1)
        training_data_res=cp.delete(training_data_res,index,0)
        index=index-1
    index=index+1










def k(x,y):
    return cp.exp(-cp.sum((x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1))**2,axis=2)/(2*lengthscale))


k_UU=k(U,U)


matrix_decode=cp.linalg.solve(k_UU,training_data_res)


def k(x,y):
    return cp.exp(-cp.mean((x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1))**2,axis=2)/(2*lengthscale))
def k_mean(mu,var,y):
    return np.sqrt(np.prod(lengthscale/(lengthscale+var.reshape(mu.shape[0],-1)),axis=1).reshape(-1,1))*cp.exp(-cp.mean((mu.reshape(mu.shape[0],1,-1)-y.reshape(1,y.shape[0],-1))**2/(2*(lengthscale+var.reshape(mu.shape[0],1,-1))),axis=2))
def k_var(mu,var,y):
    return np.sqrt(np.prod(0.5*lengthscale/(0.5*lengthscale+var.reshape(mu.shape[0],-1)),axis=1).reshape(-1,1))*cp.exp(-cp.mean((mu.reshape(mu.shape[0],1,-1)-y.reshape(1,y.shape[0],-1))**2/(2*(0.5*lengthscale+var.reshape(mu.shape[0],1,-1))),axis=2))


def decode(v):
    return k(v,U)@matrix_decode


def decode_mean(mu,sigma):
    return k(mu,U)@matrix_decode#k_mean(mu,sigma,U)@matrix_decode

def decode_var(mu,sigma):
    tmp=k_var(mu,sigma,U)- k_mean(mu,sigma,U)**2
    return cp.sum(tmp@matrix_decode**2)



U_train=U[:-1]
U_test=U[-1:]

diffs_train=U[1:]-U[:-1]


Gamma=3*cp.max(cp.max(U,axis=0)-cp.min(U,axis=0))**rank*cp.max(cp.abs(diffs_train),axis=0)

k_train=k(U_train,U_train)




k_test_train=k(U_test,U_train)
k_test=k(U_test,U_test)




k_train_reg=k_train
num_times=k_train_reg.shape[1]



matrix=cp.linalg.solve(k_train_reg,diffs_train)

Beta=cp.sqrt(Gamma**2-cp.diag(diffs_train.T@matrix))









def batched_matmul(X,Y):
    return cp.einsum('Bik,Bkj ->Bij', X, Y)


n=100




def mu(x):
    return k(x,U_train)@matrix

def sigma2(x):
    return k(x,x)-k(x,U_train)@cp.linalg.solve(k_train,(k(U_train,x)))

def f(t,y):
    mean=y[:rank]
    mean_diff=mu(mean.reshape(1,-1))
    diffusion=cp.tile(cp.diag(sigma2(mean.reshape(1,-1))).reshape(-1,1),(1,rank))
    return cp.concatenate((mean_diff.reshape(-1),diffusion.reshape(-1)))



start=cp.concatenate((U_test.reshape(-1),cp.zeros(rank)))
rec=cp.zeros((config.target_prediction_time,start.shape[0]))


rec[0]=start+f(0,start)


for i in range(1,config.target_prediction_time):
    rec[i]=rec[i-1]+f(0,rec[i-1])


data_U_test=rec[:,:rank]


std_U_test=cp.sqrt(rec[:,rank:])

std_U_test_bak=std_U_test.copy()

#std_U_test=cp.maximum(std_U_test,1.5*cp.ones_like(std_U_test))



#Cov2=cp.zeros((std_U_test.shape[0],rank**2,rank**2))
#for i in range(std_U_test.shape[0]):
#    Cov2[i]=compute_cov(cov_U_test[i])



testing_re_mean=cp.array(np.load(config.download_path+"/re_testing_mean.npy"))
#testing_re_max=np.array(np.load(config.download_path+"/re_testing_max.npy"))
#testing_re_min=np.array(np.load(config.download_path+"/re_testing_min.npy"))

data_rec_test=decode_mean(data_U_test,std_U_test**2)*r*cp.sqrt(training_data_res.shape[1])+testing_re_mean

var_rec_test=decode_var(data_U_test,std_U_test**2)*r**2*(training_data_res.shape[1])
std_rec_test=np.sqrt(var_rec_test)

perc=0.95


data_rec_test=ub*(data_rec_test>ub)+lb*(data_rec_test<lb)+(data_rec_test)*(data_rec_test<ub)*(data_rec_test>lb)



ub_rec_test=data_rec_test+3*std_rec_test
lb_rec_test=data_rec_test-3*std_rec_test


lb_rec_test[lb_rec_test<lb]=lb
ub_rec_test[ub_rec_test>ub]=ub


np.save(config.download_path+"/mean.npy",data_rec_test.get())
np.save(config.download_path+"/lb.npy",lb_rec_test.get())
np.save(config.download_path+"/ub.npy",ub_rec_test.get())

#These are needed for the theoretical validation
np.save(config.download_path+"/latent_mean.npy",data_U_test.get()) 
np.save(config.download_path+"/matrix.npy",matrix.get())
np.save(config.download_path+"/u.npy",U.get())

np.save(config.download_path+"/beta.npy",Beta.get())

end_time=time()

print(end_time-start_time)