import numpy as np
import cvxpy
from scipy.integrate import solve_ivp

def my_svd(training_data_res):
    tmp=training_data_res.T
    Omega=np.random.rand(tmp.shape[1],500)
    Y=tmp@Omega
    Q = np.linalg.qr(Y)[0]
    B = np.dot(Q.T,tmp)
    U,s,V= np.linalg.svd(B, full_matrices=False)
    U=Q.dot(U)
    return V.T,s,U.T

def predict(U,target_prediction_time,rank,rank2,t=None):

    myrank=U.shape[1]

    max_U=np.max(U)
    min_U=np.min(U)
    U=(U-min_U)/(max_U-min_U)

    U=U/np.sqrt(myrank)

    def kern(x,sigma):
        return (1+np.sqrt(5)/sigma*x+5/3*x**2/sigma**2)*np.exp(-np.sqrt(5)*x/sigma)


    def k(x,y):
        d=(np.linalg.norm(x,axis=1)**2).reshape(-1,1)+(np.linalg.norm(y,axis=1)**2).reshape(1,-1)-2*(x@y.T)
        d=np.maximum(d,np.zeros_like(d))
        d=np.sqrt(d)
        np.fill_diagonal(d,0)
        return 0.1*kern(d,np.sqrt(myrank))+0.9*kern(d,1e-08)
        
        
    def k2(x,y):
        d=((np.linalg.norm(x,axis=1)**2).reshape(-1,1)+(np.linalg.norm(y,axis=1)**2).reshape(1,-1)-2*(x@y.T))
        d=np.maximum(d,np.zeros_like(d))
        d=np.sqrt(d)
        np.fill_diagonal(d,0)
        return 0.1*kern(d,np.sqrt(myrank))+0.9*kern(d,1e-08)


    if t is None:
        t=np.arange(U.shape[0])

    U_train=U[:-1]
    U_test=U[-1:]
    tmp=np.arange(U.shape[1])
    diffs_train=np.gradient(U,t,tmp)[0][:-1]
    X_train=U[:-1]
    k_train=k(X_train,X_train)
    k_train_2=k2(X_train,X_train)
    k_train_reg=k_train
    matrix=np.linalg.solve(k_train_reg,diffs_train)


    Z_var=np.max(np.var(U_train,axis=0))

    def mu(x):
        x=x.reshape(1,-1)
        return k(x,U_train)@matrix

    def sigma2(x):
        return (k2(x,x)-k2(x,U_train)@np.linalg.solve(k_train_2,(k2(U_train,x))))



    def f(t,y):
        mean=y[:myrank]
        prev_cov=y[myrank:]
        mean_diff=mu(mean.reshape(1,-1)).reshape(-1)
        diffusion=np.tile(np.diag(sigma2(mean.reshape(1,-1))).reshape(-1,1),(1,myrank))
        new_cov=diffusion.reshape(-1)*(Z_var-prev_cov)
        return np.concatenate((mean_diff,new_cov))


    start=np.concatenate((U_test.reshape(-1),np.zeros(myrank)))
    t_span = (0, target_prediction_time)
    rec = solve_ivp(f, t_span, start, t_eval=np.linspace(0, target_prediction_time, target_prediction_time+1)).y.T


    if t is None:
        t=np.arange(U.shape[0])
    

    
    


    data_U_test=rec[1:,:myrank]
    var_U_test=(rec[1:,myrank:2*myrank])


    data_U_test=data_U_test*np.sqrt(myrank)
    data_U_test=data_U_test*(max_U-min_U)+min_U
    var_U_test=var_U_test*(max_U-min_U)**2


    return data_U_test, var_U_test, 0, 0
