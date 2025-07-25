import numpy as np
from staticvariables import *
import sys
from time import time
config=__import__(sys.argv[1])
import cupy as cp





start_time=time()
lb=variables_dict[config.variable][0]
ub=variables_dict[config.variable][1]

training_re_mean=cp.array(np.load(config.download_path+"/re_training_mean.npy"))

training_data=cp.array(np.load(config.download_path+"/training_data.npy"))

def compute_cov(C):
    a = cp.arange(rank2)
    b = cp.arange(rank2)

    # Create the NxN matrix where A[i, j, 0] = a[i] and A[i, j, 1] = b[j]
    A = cp.zeros((len(a), len(b), 2))

    # Populate A[i, j, 0] = a[i] and A[i, j, 1] = b[j]
    A[:, :, 0] = a[:, cp.newaxis]  # broadcasting a[i]
    A[:, :, 1] = b[cp.newaxis, :]  # broadcasting b[j]


    A=A.reshape(-1,2)

    B = cp.zeros((len(A), len(A), 4),dtype=cp.int64)
    B[:,:,:2]=A[:,cp.newaxis]
    B[:,:,2:]=A[cp.newaxis:]
    B=B.reshape(rank2,rank2,rank2,rank2,-1)
    B0=B[:,:,:,:,0]
    B1=B[:,:,:,:,1]
    B2=B[:,:,:,:,2]
    B3=B[:,:,:,:,3]
    S=C[B0,B1]*C[B2,B3]+C[B0,B2]*C[B1,B3]+C[B0,B3]*C[B1,B2]
    return S.reshape(rank2**2,rank2**2)



training_data_res=training_data-training_re_mean

training_data_mean=cp.mean(training_data_res,axis=0)
training_data_res=training_data_res-training_data_mean

start=time()

def my_svd(training_data_res):
    tmp=training_data_res.T
    Omega=cp.random.rand(tmp.shape[1],500)
    Y=tmp@Omega
    Q = cp.linalg.qr(Y)[0]
    print(Q.shape)
    print(tmp.shape)
    B = cp.dot(Q.T,tmp)
    U,s,V= cp.linalg.svd(B, full_matrices=False)
    U=Q.dot(U)
    return V.T,s,U.T

U,S,V=my_svd(training_data_res)#cp.linalg.svd(training_data_res,full_matrices=False)

end=time()
print(end-start)


S=S/cp.sqrt(len(training_data))

rank=cp.argmin(cp.diff(cp.diff(S)))
rank=cp.maximum(rank,100*cp.ones_like(rank)).item()
rank2=2




def solve_constr(A,B,C,a):
    return cp.linalg.solve((B.T@(B@C)@C.T+a*cp.eye(rank2**2)),B.T@A@C.T)


S2=S[rank:rank+rank2**2]
V2=V[rank:rank+rank2**2]

S=S[:rank]
U=U[:,:rank]
V=V[:rank]
U=cp.sqrt(len(training_data_res))*U
W=cp.diag(S)@V

U_quad=U[:,:rank2]

Winv=V.T@cp.diag(1/(S))

W2=0
U2=0


err_train=training_data_res-U@W

flag2=False

if S2.shape[0]== rank2**2:
    flag2=True
    W2=cp.zeros((rank2**2,training_data_mean.shape[0]))
    U2=cp.zeros((U.shape[0],rank2**2))


    M=cp.diag(S2)@V2

    err=U@W-training_data_res
    err_std=cp.std(err,axis=0)



    B=cp.concatenate((err,M),axis=0)


    
    E2=cp.eye(rank2).reshape(-1)
    Cov2=compute_cov(cp.eye(rank2**2))





    U2=U_quad.reshape(U.shape[0],1,-1)*U_quad.reshape(U.shape[0],-1,1)
    U2=U2.reshape(U.shape[0],rank2**2)


    Pmu=cp.eye(rank2**2)-E2.reshape(-1,1)@E2.reshape(1,-1)/(cp.dot(E2,E2))

    
    u,s,v=cp.linalg.svd(Cov2)
    L=u@cp.diag(s)

    A=cp.concatenate((U2,L),axis=0)

    A=A@Pmu



    #solves ||A-BXC||+a||X||



    _,a,_=cp.linalg.svd(A.T@A,full_matrices=False)

    W2=Pmu@solve_constr(B,A,V2,a+1)@V2
    err_train=training_data_res-(U2)@W2





len_I=training_data_res.shape[0]-1

var_train=cp.var(err_train,axis=0)


del training_data
del training_data_res
del err_train


def k(x,y):
    return 0.1*cp.exp(-cp.mean(cp.abs(x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1)),axis=2)/(2))+0.9*cp.exp(-cp.mean(cp.abs(x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1)),axis=2)/(2*(1e-08)))

def k2(x,y):
    return 0.1*cp.exp(-cp.mean(cp.abs(x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1)),axis=2)/(2))+0.9*cp.exp(-cp.mean(cp.abs(x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1)),axis=2)/(2*(1e-08)))


U_train=U[:-1]
U_test=U[-1:]

diffs_train=U[1:]-U[:-1]


Gamma=3*cp.max(cp.max(U,axis=0)-cp.min(U,axis=0))**rank*cp.max(cp.abs(diffs_train),axis=0)

k_train=k(U_train,U_train)


k_train_2=k2(U_train,U_train)


k_test_train=k(U_test,U_train)
k_test=k(U_test,U_test)

k_test_train_2=k2(U_test,U_train)
k_test_2=k2(U_test,U_test)



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
    return k2(x,x)-k2(x,U_train)@cp.linalg.solve(k_train_2,(k2(U_train,x)))

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

std_U_test=cp.maximum(std_U_test,1.5*cp.ones_like(std_U_test))


cov_U_test=(std_U_test[:,:rank2]**2).reshape(data_U_test.shape[0],rank2,1)*cp.eye(rank2).reshape(1,rank2,rank2)

Cov2=0
if flag2:
    Cov2=cp.zeros((std_U_test.shape[0],rank2**2,rank2**2))    
    for i in range(std_U_test.shape[0]):
        Cov2[i]=compute_cov(cov_U_test[i])



testing_re_mean=cp.array(np.load(config.download_path+"/re_testing_mean.npy"))
#testing_re_max=np.array(np.load(config.download_path+"/re_testing_max.npy"))
#testing_re_min=np.array(np.load(config.download_path+"/re_testing_min.npy"))

data_U2_test=data_U_test[:,:rank2].reshape(config.target_prediction_time,1,-1)*data_U_test[:,:rank2].reshape(config.target_prediction_time,-1,1)
data_rec_test=data_U_test@W+training_data_mean+testing_re_mean

if flag2:
    data_rec_test=data_rec_test+(data_U2_test.reshape(-1,rank2**2))@W2


var_rec_test=cp.sum(std_U_test**2@W**2)+var_train

if flag2:
    var_rec_test=var_rec_test+cp.sum((cp.einsum('aij,ajk->aik',Cov2,W2.reshape(1,rank2**2,-1))*W2.reshape(1,rank2**2,-1)),axis=1)

std_rec_test=cp.sqrt(var_rec_test)

perc=0.95


data_rec_test=ub*(data_rec_test>ub)+lb*(data_rec_test<lb)+(data_rec_test)*(data_rec_test<ub)*(data_rec_test>lb)



ub_rec_test=data_rec_test+3*std_rec_test
lb_rec_test=data_rec_test-3*std_rec_test


lb_rec_test[lb_rec_test<lb]=lb
ub_rec_test[ub_rec_test>ub]=ub


np.save(config.download_path+"/mean.npy",data_rec_test.get())
np.save(config.download_path+"/lb.npy",lb_rec_test.get())
np.save(config.download_path+"/ub.npy",ub_rec_test.get())
np.save(config.download_path+"/train_mean.npy",training_data_mean.get())

#These are needed for the theoretical validation
np.save(config.download_path+"/latent_mean.npy",data_U_test.get()) 
np.save(config.download_path+"/latent_var.npy",cov_U_test.get())
np.save(config.download_path+"/winv.npy",Winv.get())
np.save(config.download_path+"/matrix.npy",matrix.get())
np.save(config.download_path+"/u.npy",U.get())

np.save(config.download_path+"/beta.npy",Beta.get())

end_time=time()

print(end_time-start_time)