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






training_data_res=training_data-training_re_mean
training_data_mean=cp.mean(training_data_res,axis=0)
training_data_res=training_data_res-training_data_mean



training_data_res=training_data_res/cp.sqrt(training_data_res.shape[1])


r=cp.sqrt(cp.mean(cp.linalg.norm(training_data_res,axis=1)**2))
training_data_res=training_data_res/r




testing_data=training_data_res[-1].reshape(1,-1)

training_data_res_diff=training_data_res[1:]-training_data_res[:-1]
training_data_res=training_data_res[:-1]

tmp=cp.sum(training_data_res**2,axis=1).reshape(-1,1)+cp.sum(training_data_res**2,axis=1).reshape(1,-1)-2*cp.dot(training_data_res,training_data_res.T)


tmp_2=tmp.copy()
cp.fill_diagonal(tmp_2,999)


lengthscale=10*cp.mean((cp.sort(tmp_2,axis=1))[:,0])/2

index=0

while index<len(tmp):
    if cp.sum(tmp_2[index]<0.05*lengthscale)>0:
        training_data_res_diff=cp.delete(training_data_res_diff,index,axis=0)
        training_data_res=cp.delete(training_data_res,index,axis=0)
        tmp_2=cp.delete(tmp_2,index,0)
        tmp_2=cp.delete(tmp_2,index,1)
        tmp=cp.delete(tmp,index,0)
        tmp=cp.delete(tmp,index,1)

        index=index-1
    index=index+1


def k(x,y):
    return cp.exp(-(cp.sum(x**2,axis=1).reshape(-1,1)+cp.sum(y**2,axis=1).reshape(1,-1)-2*cp.dot(x,y.T))/(2*lengthscale))


k_train=k(training_data_res,training_data_res)

k_test_train=k(testing_data,training_data_res)
k_test=k(testing_data,testing_data)



matrix=cp.linalg.solve(k_train,training_data_res_diff)

   
Gamma=cp.sqrt(4*cp.sum(training_data_res_diff*matrix,axis=0))


Beta=cp.sqrt(Gamma**2-cp.sum(training_data_res_diff*matrix,axis=0))





data_rec_test=cp.zeros((config.target_prediction_time,training_data.shape[1]))

n=100




def mu(x):
    return k(x,training_data_res)@matrix





start=testing_data

print(cp.min(training_data_res),cp.max(training_data_res))

print(cp.min(testing_data),cp.max(testing_data))

data_rec_test[0]=mu(testing_data)

for i in range(1,config.target_prediction_time):
    data_rec_test[i]=data_rec_test[i-1]+mu(data_rec_test[i-1].reshape(1,-1)).reshape(-1)


print(cp.min(data_rec_test),cp.max(data_rec_test))


data_rec_test=data_rec_test*r*cp.sqrt(training_data_res.shape[1])

del tmp
del tmp_2
del training_data


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






def my_svd(training_data_res):
    tmp=training_data_res.T
    Omega=cp.random.rand(tmp.shape[1],500)
    Y=tmp@Omega
    Q = cp.linalg.qr(Y)[0]
    B = cp.dot(Q.T,tmp)
    U,s,V= cp.linalg.svd(B, full_matrices=False)
    U=Q.dot(U)
    return V.T,s,U.T

training_data_res=training_data_res*r*cp.sqrt(training_data_res.shape[1])

U,S,V=my_svd(training_data_res)#cp.linalg.svd(training_data_res,full_matrices=False)


end=time()


S=S/cp.sqrt(len(training_data_res))

rank=cp.argmin(cp.diff(cp.diff(S)))
rank=cp.maximum(rank,2*cp.ones_like(rank)).item()





def solve_constr(A,B,C,a):
    return cp.linalg.solve((B.T@(B@C)@C.T+a*cp.eye(rank**2)),B.T@A@C.T)


S2=S[rank:rank+rank**2]
V2=V[rank:rank+rank**2]

S=S[:rank]
U=U[:,:rank]
V=V[:rank]
U=cp.sqrt(len(training_data_res))*U
W=cp.diag(S)@V


Winv=V.T@cp.diag(1/(S))




W2=0#cp.zeros((rank**2,training_data_mean.shape[0]))
U2=0#cp.zeros((U.shape[0],rank**2))


flag_square=False

err_train=training_data_res-U@W


if S2.shape[0]==rank**2:
    W2=cp.zeros((rank**2,training_data_mean.shape[0]))
    U2=cp.zeros((U.shape[0],rank**2))
    flag_square=True
    M=cp.diag(S2)@V2

    err=U@W-training_data_res
    err_std=cp.std(err,axis=0)



    B=cp.concatenate((err,M),axis=0)


    
    E2=cp.eye(rank).reshape(-1)
    Cov2=compute_cov(cp.eye(rank))





    U2=U.reshape(U.shape[0],1,-1)*U.reshape(U.shape[0],-1,1)
    U2=U2.reshape(U.shape[0],rank**2)


    Pmu=cp.eye(rank**2)-E2.reshape(-1,1)@E2.reshape(1,-1)/(cp.dot(E2,E2))

    
    u,s,v=cp.linalg.svd(Cov2)
    L=u@cp.diag(s)

    A=cp.concatenate((U2,L),axis=0)

    A=A@Pmu



    #solves ||A-BXC||+a||X||



    _,a,_=cp.linalg.svd(A.T@A,full_matrices=False)

    W2=Pmu@solve_constr(B,A,V2,a+1)@V2

    del V2
    del Pmu
    del B
    del A

    err_train=err_train-(U2)@W2

var_train=cp.var(err_train,axis=0)





U_train=U[:-1]
U_test=U[-1:]

diffs_train=U[1:]-U[:-1]


len_I=training_data_res.shape[0]-1


lengthscale=1
tmp=cp.sum((U_train.reshape(U_train.shape[0],1,rank)-U_train.reshape(1,U_train.shape[0],rank))**2,axis=2)

tmp_2=tmp.copy()
cp.fill_diagonal(tmp_2,999)


lengthscale_lat=cp.mean((cp.sort(tmp_2,axis=1))[:,0])



lengthscale_lat=cp.minimum(cp.array(1.),lengthscale_lat+1e-08)/2





cp.fill_diagonal(tmp,999)

index=0


while index<len(tmp):
    if cp.sum(tmp[index]<0.2*lengthscale)>0:
        U_train=cp.delete(U_train,index,axis=0)
        tmp=cp.delete(tmp,index,0)
        tmp=cp.delete(tmp,index,1)
        index=index-1
    index=index+1








def k2(x,y):
    return cp.exp(-cp.mean((x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1))**2,axis=2)/(2*lengthscale_lat))


data_U_test=data_rec_test@Winv


k_train_2=k2(U_train,U_train)


k_test_train_2=k2(U_test,U_train)
k_test_2=k2(U_test,U_test)


var_U_test=cp.zeros((config.target_prediction_time,rank))


def sigma2(x):
    return k2(x,x)-k2(x,U_train)@cp.linalg.solve(k_train_2,(k2(U_train,x)))


var_U_test[0]=sigma2(testing_data@Winv)


for i in range(1,config.target_prediction_time):
    var_U_test[i]=var_U_test[i-1]+sigma2(data_U_test[i-1].reshape(1,-1))


std_U_test=cp.sqrt(var_U_test)

#std_U_test_bak=std_U_test.copy()

std_U_test=cp.minimum(std_U_test,3*cp.ones_like(std_U_test))




Cov2=0
cov_U_test=cp.zeros([0])

if flag_square:
    cov_U_test=(std_U_test**2).reshape(std_U_test.shape[0],rank,1)*cp.eye(rank).reshape(1,rank,rank)
    Cov2=cp.zeros((std_U_test.shape[0],rank**2,rank**2))
    for i in range(std_U_test.shape[0]):
        Cov2[i]=compute_cov(cov_U_test[i])



testing_re_mean=cp.array(np.load(config.download_path+"/re_testing_mean.npy"))


data_rec_test=data_rec_test+training_data_mean+testing_re_mean


var_rec_test=(std_U_test**2)@(W**2)

del training_data_res
del training_data_res_diff
del W


if flag_square:
    data_U2_test=data_U_test.reshape(config.target_prediction_time,1,-1)*data_U_test.reshape(config.target_prediction_time,-1,1)
    var_rec_test=var_rec_test+cp.sum((cp.einsum('aij,ajk->aik',Cov2,W2.reshape(1,rank**2,-1))*W2.reshape(1,rank**2,-1)),axis=1)

std_rec_test=cp.sqrt(var_rec_test+var_train)

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