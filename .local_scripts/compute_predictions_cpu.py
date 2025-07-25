import numpy as np
from staticvariables import *
import sys
from time import time
import importlib.util
from qgprsde import predict
def module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("miao", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config=module_from_file(sys.argv[1])






start_time=time()
lb=variables_dict[config.variable][0]
ub=variables_dict[config.variable][1]

training_re_mean=np.load(config.download_path+"/re_training_mean.npy")

training_data=np.load(config.download_path+"/training_data.npy")

rank=100
rank2=2

data_rec_test, var_rec_test, err_train, var_train=predict(training_data-training_re_mean,config.target_prediction_time,rank,rank2)
testing_re_mean=(np.load(config.download_path+"/re_testing_mean.npy"))
data_rec_test=data_rec_test+testing_re_mean

'''
def compute_cov(C):
    a = np.arange(rank2)
    b = np.arange(rank2)

    # Create the NxN matrix where A[i, j, 0] = a[i] and A[i, j, 1] = b[j]
    A = np.zeros((len(a), len(b), 2))

    # Populate A[i, j, 0] = a[i] and A[i, j, 1] = b[j]
    A[:, :, 0] = a[:, np.newaxis]  # broadcasting a[i]
    A[:, :, 1] = b[np.newaxis, :]  # broadcasting b[j]


    A=A.reshape(-1,2)

    B = np.zeros((len(A), len(A), 4),dtype=np.int64)
    B[:,:,:2]=A[:,np.newaxis]
    B[:,:,2:]=A[np.newaxis:]
    B=B.reshape(rank2,rank2,rank2,rank2,-1)
    B0=B[:,:,:,:,0]
    B1=B[:,:,:,:,1]
    B2=B[:,:,:,:,2]
    B3=B[:,:,:,:,3]
    S=C[B0,B1]*C[B2,B3]+C[B0,B2]*C[B1,B3]+C[B0,B3]*C[B1,B2]
    return S.reshape(rank2**2,rank2**2)



training_data_res=training_data-training_re_mean

training_data_mean=np.mean(training_data_res,axis=0)
training_data_res=training_data_res-training_data_mean



def my_svd(training_data_res,rank):

    B=training_data_res@training_data_res.T
    U,s,V= np.linalg.svd(B, full_matrices=False)
    U=U[:,:rank]
    return U,training_data_res.T@(np.linalg.inv(B)@U)


U,Winv=my_svd(training_data_res, 100)#np.linalg.svd(training_data_res,full_matrices=False)








U=U/np.linalg.norm(U,axis=1).reshape(-1,1)


W=(np.linalg.inv(U.T@U)@U.T)@training_data_res


U_quad=U[:,:rank2]

U2=U_quad.reshape(U.shape[0],1,-1)*U_quad.reshape(U.shape[0],-1,1)

U2=U2.reshape(U.shape[0],-1)

err_train=training_data_res-U@W


a=np.linalg.norm(U2.T@U2)

W2=(np.linalg.inv(U2.T@U2+a*np.eye(U2.shape[1]))@U2.T)@err_train


err_train=training_data_res-U@W-U2@W2


len_I=training_data_res.shape[0]-1

var_train=np.var(err_train,axis=0)



del training_data
del training_data_res
del err_train


#def k(x,y):
#    return 0.1*np.exp(-np.mean(np.abs(x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1)),axis=2)/(2))+0.9*np.exp(-np.mean(np.abs(x.reshape(x.shape[0],1,-1)-y.reshape(1,y.shape[0],-1)),axis=2)/(2*(1e-08)))



def k(x,y):
    d=(np.linalg.norm(x,axis=1)**2).reshape(-1,1)+(np.linalg.norm(y,axis=1)**2).reshape(1,-1)-2*(x@y.T)
    return (1*np.exp(-(d)/(2*(rank+1)))+1*np.exp(-(d)/(2*1e-08)))
    
    
def k2(x,y):
    d=(np.linalg.norm(x,axis=1)**2).reshape(-1,1)+(np.linalg.norm(y,axis=1)**2).reshape(1,-1)-2*(x@y.T)
    return (1*np.exp(-(d)/(2*(rank+1)))+1*np.exp(-(d)/(2*1e-08)))



U_train=U[:-1]
U_test=U[-1:]

diffs_train=U[1:]-U[:-1]

X_train=U[:-1]



k_train=k(X_train,X_train)

np.fill_diagonal(k_train,1)

k_train_2=k2(X_train,X_train)


k_test_train=k(U_test,U_train)
k_test=k(U_test,U_test)

k_test_train_2=k2(U_test,U_train)
k_test_2=k2(U_test,U_test)



k_train_reg=k_train
num_times=k_train_reg.shape[1]



matrix=np.linalg.solve(k_train_reg,diffs_train)










def batched_matmul(X,Y):
    return np.einsum('Bik,Bkj ->Bij', X, Y)


n=100




def mu(x):
    x=x.reshape(1,-1)
    return (k(x,U_train)@matrix@(np.eye(rank)-x.reshape(-1,1)@x.reshape(1,-1)/(np.linalg.norm(x)**2))).reshape(-1)

def sigma2(x):
    return k2(x,x)-k2(x,U_train)@np.linalg.solve(k_train_2,(k2(U_train,x)))



matrix_npu=matrix
U_train_npu=U_train


def f(t,y):
    mean=y[:rank]
    prev_cov=y[rank:2*rank]
    prev_m3=y[2*rank:3*rank]
    prev_m4=y[3*rank:4*rank]
    mean_diff=mu(mean.reshape(1,-1))

    new_mean=mean+mean_diff.reshape(-1)
    new_mean=new_mean/np.linalg.norm(new_mean)
    diffusion=np.tile(np.diag(sigma2(new_mean.reshape(1,-1))).reshape(-1,1),(1,rank))
    new_cov=prev_cov+diffusion.reshape(-1)
    diffusion=diffusion.reshape(-1)
    new_m3=prev_m3+3*new_cov*new_mean
    new_m4=prev_m4+6*diffusion*new_cov+4*new_m3*new_mean#*np.maximum(new_m3*new_mean,np.zeros(rank))
    return np.concatenate((new_mean,new_cov,new_m3,new_m4))



start=np.concatenate((U_test.reshape(-1),np.zeros(rank),np.zeros(rank),np.zeros(rank)))
rec=np.zeros((config.target_prediction_time,start.shape[0]))


rec[0]=f(0,start)



for i in range(1,config.target_prediction_time):
    rec[i]=f(0,rec[i-1])






data_U_test=rec[:,:rank]


var_U_test=(rec[:,rank:2*rank])


coeff=np.sqrt(np.mean(var_U_test[0])/np.mean(np.var(U,axis=0)))*(5)

var_U_test=var_U_test/(coeff**2)
m3_U_test=rec[:,2*rank:3*rank]/(coeff**3)
m4_U_test=rec[:,3*rank:4*rank]/(coeff**4)

E2_U_test=var_U_test+data_U_test**2
E3_U_test=m3_U_test+data_U_test**3+3*data_U_test*var_U_test
E4_U_test=m4_U_test+data_U_test**4+4*data_U_test*m3_U_test+6*data_U_test**2*var_U_test


def compute_outer_product_covariance(moments):
    """
    Given a (n x 4) matrix 'moments' where each row i contains:
       moments[i,0] = E[Z_i]    (mean)
       moments[i,1] = E[Z_i^2]  (second moment)
       moments[i,2] = E[Z_i^3]  (third moment)
       moments[i,3] = E[Z_i^4]  (fourth moment)
    this function computes the covariance matrix of the outer product Z ⊗ Z.
    
    The resulting covariance matrix is of shape (n^2, n^2) where the entry indexed
    by ((i,j), (k,l)) is given by:
    
       Cov(Z_i Z_j, Z_k Z_l) = E[Z_i Z_j Z_k Z_l] - E[Z_i Z_j] E[Z_k Z_l].
       
    E[Z_i Z_j] is taken to be moments[i,1] if i == j, otherwise moments[i,0]*moments[j,0].
    The fourth-order expectation is computed by noting that due to independence:
    
       E[Z_i Z_j Z_k Z_l] = ∏_{r in {i,j,k,l}} M_r,
       
    where if an index appears exactly r times then M_r is:
         r == 1:  moments[idx, 0]
         r == 2:  moments[idx, 1]
         r == 3:  moments[idx, 2]
         r == 4:  moments[idx, 3]
    
    Parameters:
      moments (np.ndarray): An (n x 4) array of moments.
    
    Returns:
      cov (np.ndarray): A (n^2 x n^2) covariance matrix.
    """
    n = moments.shape[0]
    
    # Precompute pairwise expectations E[Z_i Z_j]
    EZ = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                EZ[i, j] = moments[i, 1]  # second moment when i == j
            else:
                EZ[i, j] = moments[i, 0] * moments[j, 0]  # product of means for i != j
                
    def fourth_moment(i, j, k, l):
        """
        Computes E[Z_i Z_j Z_k Z_l] for independent random variables by
        multiplying the appropriate moments based on index multiplicities.
        """
        indices = [i, j, k, l]
        counts = {}
        for idx in indices:
            counts[idx] = counts.get(idx, 0) + 1
        # Multiply the corresponding moments
        result = 1.0
        for idx, cnt in counts.items():
            if cnt == 1:
                result *= moments[idx, 0]
            elif cnt == 2:
                result *= moments[idx, 1]
            elif cnt == 3:
                result *= moments[idx, 2]
            elif cnt == 4:
                result *= moments[idx, 3]
        return result


    def third_moment(i, j, k):
        """
        Computes E[Z_i Z_j Z_k] for independent random variables by
        multiplying the appropriate moments based on index multiplicities.
        """
        indices = [i, j, k]
        counts = {}
        for idx in indices:
            counts[idx] = counts.get(idx, 0) + 1
        # Multiply the corresponding moments
        result = 1.0
        for idx, cnt in counts.items():
            if cnt == 1:
                result *= moments[idx, 0]
            elif cnt == 2:
                result *= moments[idx, 1]
            elif cnt == 3:
                result *= moments[idx, 2]
        return result


    cov_single=moments[:,1]-np.diag(moments[:,0]**2)

    # Build the covariance matrix for the vectorized outer product (of size n^2)
    cov = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            row = i * n + j
            for k in range(n):
                for l in range(n):
                    col = k * n + l
                    exp4 = fourth_moment(i, j, k, l)
                    cov[row, col] = exp4 - EZ[i, j] * EZ[k, l]

    cov_mixed=np.zeros((n*n, n))
    for i in range(n):
        for j in range(n):
            row = i * n + j
            for k in range(n):
                exp3 = third_moment(i, j, k)
                cov_mixed[row, k] = exp3- EZ[i, j] * moments[k,0]



    return EZ.reshape(-1), cov, cov_mixed


all_mean=np.zeros((config.target_prediction_time,rank+rank2**2))
all_cov=np.zeros((config.target_prediction_time,rank+rank2**2,rank+rank2**2))


var_rec_test=np.zeros((config.target_prediction_time,training_data_mean.shape[0]))


W2=np.zeros_like(W2)
W_tot=np.concatenate((W,W2),axis=0)


for i in range(config.target_prediction_time):
	tmp_mean,tmp_cov,tmp_mixed=compute_outer_product_covariance(np.concatenate((data_U_test[0,:rank2].reshape(-1,1),E2_U_test[0,:rank2].reshape(-1,1),E3_U_test[0,:rank2].reshape(-1,1),E4_U_test[0,:rank2].reshape(-1,1)),axis=1))
	all_mean[i,:rank]=data_U_test[i]
	all_mean[i,rank:]=tmp_mean
	

	all_cov[np.ix_([i],np.arange(rank),np.arange(rank))]=np.diag(var_U_test[i])
	
	all_cov[np.ix_([i],np.arange(rank,rank+rank2**2),np.arange(rank,rank+rank2**2))]=np.diag(tmp_cov)
	
	all_cov[np.ix_([i],np.arange(rank,rank+rank2**2),np.arange(0,rank2))]=tmp_mixed
	
	
	all_cov[np.ix_([i],np.arange(0,rank2),np.arange(rank,rank+rank2**2))]=tmp_mixed.T
	
	var_rec_test[i]=np.einsum('ki,kk,ki->i',W_tot,all_cov[i],W_tot)+var_train 


std_rec_test=np.sqrt(var_rec_test)



testing_re_mean=(np.load(config.download_path+"/re_testing_mean.npy"))



data_rec_test=training_data_mean+all_mean@W_tot+testing_re_mean#+data_U_test@W#+np.zeros_like(all_mean@W_tot)





perc=0.95


data_rec_test=ub*(data_rec_test>ub)+lb*(data_rec_test<lb)+(data_rec_test)*(data_rec_test<ub)*(data_rec_test>lb)


'''


std_rec_test=np.sqrt(var_rec_test)


ub_rec_test=data_rec_test+5*std_rec_test
lb_rec_test=data_rec_test-5*std_rec_test



#lb_rec_test[lb_rec_test<lb]=lb
#ub_rec_test[ub_rec_test>ub]=ub




np.save(config.download_path+"/mean.npy",data_rec_test)
np.save(config.download_path+"/lb.npy",lb_rec_test)
np.save(config.download_path+"/ub.npy",ub_rec_test)
#np.save(config.download_path+"/train_mean.npy",training_data_mean)
#These are needed for the theoretical validation
#np.save(config.download_path+"/latent_mean.npy",data_U_test)
#np.save(config.download_path+"/latent_var.npy",var_U_test)
#np.save(config.download_path+"/matrix.npy",matrix)
#np.save(config.download_path+"/u.npy",U)

end_time=time()

print(end_time-start_time)
