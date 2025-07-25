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

def predict(training_data,target_prediction_time,rank,rank2,t=None):
    training_data_mean=np.mean(training_data,axis=0)
    training_data_res=training_data-training_data_mean
    U,S,V=my_svd(training_data_res)
    S=S[:rank]
    V=V[:rank]
    U=U[:,:rank]
    M = cvxpy.Variable((rank, rank), PSD=True)
    objective = cvxpy.sum_squares(cvxpy.hstack([cvxpy.quad_form(U[i].reshape(-1), M) - 1 for i in range(U.shape[0])]))+1e-8*(cvxpy.lambda_max(M) - cvxpy.lambda_min(M))
    prob = cvxpy.Problem(cvxpy.Minimize(objective))
    prob.solve()
    M=M.value
    _,eigenvalues,eigenvectors = np.linalg.svd(M)
    B=np.sqrt(np.diag(eigenvalues))@eigenvectors
    new_rank=eigenvalues[eigenvalues>1e-04*eigenvalues[0]].shape[0]
    eigenvalues=eigenvalues[:new_rank]
    eigenvectors=eigenvectors[:new_rank]
    B_var=np.sqrt(np.diag(eigenvalues))@eigenvectors
    Z_var=U@B_var.T
    Winv=V.T@np.diag(1/S)@B.T
    Z=U@B.T
    Z=Z/np.linalg.norm(Z,axis=1).reshape(-1,1)
    B_inv=np.linalg.pinv(Z)@U
    W=B_inv@np.diag(S)@V
    B_var_inv=np.linalg.pinv(Z_var)@U
    B_var_inv=np.concatenate((B_var_inv,np.zeros((rank-new_rank,B.shape[1]))),axis=0)
    W_var=B_var_inv@np.diag(S)@V
    U=Z
    U_quad=U[:,:rank2]
    U2=U_quad.reshape(U.shape[0],1,-1)*U_quad.reshape(U.shape[0],-1,1)
    U2=U2.reshape(U.shape[0],-1)
    err_train=training_data_res-U@W
    a=np.linalg.norm(U2.T@U2)
    W2=np.linalg.inv(U2.T@U2+a*np.eye(U2.shape[1]))@U2.T@err_train
    W2=W2-W2@Winv@np.linalg.solve((Winv.T@Winv),Winv.T)
    err_train=training_data_res-U@W-U2@W2
    err_train_mean=np.mean(err_train)
    var_train=np.var(err_train,axis=0)


    def kern(x,sigma):
        return (1+np.sqrt(3)/sigma*x)*np.exp(-np.sqrt(3)*x/sigma)

        return #(1+np.sqrt(5)/sigma*x+5/3*x**2/sigma**2)*np.exp(-np.sqrt(5)*x/sigma)


    def k(x,y):
        d=(np.linalg.norm(x,axis=1)**2).reshape(-1,1)+(np.linalg.norm(y,axis=1)**2).reshape(1,-1)-2*(x@y.T)
        d=np.maximum(d,np.zeros_like(d))
        d=np.minimum(d,np.ones_like(d))
        d=np.sqrt(d)
        return 0.1*kern(d,rank+1)+0.9*kern(d,1e-08)
        
        
    def k2(x,y):
        d=((np.linalg.norm(x,axis=1)**2).reshape(-1,1)+(np.linalg.norm(y,axis=1)**2).reshape(1,-1)-2*(x@y.T))
        d=np.maximum(d,np.zeros_like(d))
        d=np.minimum(d,np.ones_like(d))
        d=np.sqrt(d)
        return 0.1*kern(d,rank+1)+0.9*kern(d,1e-08)



    U_train=U[:-1]
    U_test=U[-1:]
    #diffs_train=U[1:]-U[:-1]

    if t is None:
        t=np.arange(U.shape[0])
    tmp=np.arange(U.shape[1])
    diffs_train=np.gradient(U,t,tmp)[0][:-1]
    X_train=U[:-1]
    k_train=k(X_train,X_train)
    np.fill_diagonal(k_train,1)
    k_train_2=k2(X_train,X_train)
    k_train_reg=k_train
    matrix=np.linalg.solve(k_train_reg,diffs_train)

    Z_var=np.max(np.var(X_train,axis=0))
 

    def mu(x):
        x=x.reshape(1,-1)
        return (k(x,U_train)@matrix@(np.eye(rank)-x.reshape(-1,1)@x.reshape(1,-1)/(np.linalg.norm(x)**2))).reshape(-1)

    def sigma2(x):
        tmp=(k2(x,x)-k2(x,U_train)@np.linalg.solve(k_train_2,(k2(U_train,x))))
        tmp=np.maximum(tmp,0)
        tmp=(tmp>1)*tmp+2*tmp**2/(1+tmp**2)*(tmp<1)
        tmp=tmp
        return tmp



    def f(t,y):
        mean=y[:rank]
        prev_cov=y[rank:2*rank]
        prev_m3=y[2*rank:3*rank]
        prev_m4=y[3*rank:4*rank]
        mean_diff=mu(mean.reshape(1,-1))
        diffusion=np.tile(np.diag(sigma2(mean.reshape(1,-1))).reshape(-1,1),(1,rank))*(Z_var-prev_cov)
        new_cov=diffusion.reshape(-1)
        new_m3=3*prev_cov*mean_diff
        new_m4=(6*diffusion*prev_cov+4*prev_m3*mean).reshape(-1)*(2*Z_var**2-prev_m4)
        return np.concatenate((mean_diff,new_cov,new_m3,new_m4))



    start=np.concatenate((U_test.reshape(-1),np.zeros(rank),np.zeros(rank),np.zeros(rank)))


    t_span = (0, target_prediction_time)
    rec = solve_ivp(f, t_span, start, t_eval=np.linspace(0, target_prediction_time, target_prediction_time+1)).y.T
    data_U_test=rec[:,:rank]
    var_U_test=(rec[:,rank:2*rank])
    m3_U_test=rec[:,2*rank:3*rank]
    m4_U_test=rec[:,3*rank:4*rank]
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


    all_mean=np.zeros((target_prediction_time,rank+rank2**2))
    all_cov=np.zeros((target_prediction_time,rank+rank2**2,rank+rank2**2))


    var_rec_test=np.zeros((target_prediction_time,training_data_mean.shape[0]))


    W_tot=np.concatenate((W,W2),axis=0)
    W_tot_var=np.concatenate((W_var,W2),axis=0)


    for i in range(target_prediction_time):
        tmp_mean,tmp_cov,tmp_mixed=compute_outer_product_covariance(np.concatenate((data_U_test[0,:rank2].reshape(-1,1),E2_U_test[0,:rank2].reshape(-1,1),E3_U_test[0,:rank2].reshape(-1,1),E4_U_test[0,:rank2].reshape(-1,1)),axis=1))
        all_mean[i,:rank]=data_U_test[i]
        all_mean[i,rank:]=tmp_mean
        all_cov[np.ix_([i],np.arange(rank),np.arange(rank))]=np.diag(var_U_test[i])
        all_cov[np.ix_([i],np.arange(rank,rank+rank2**2),np.arange(rank,rank+rank2**2))]=np.diag(tmp_cov)
        all_cov[np.ix_([i],np.arange(rank,rank+rank2**2),np.arange(0,rank2))]=tmp_mixed

        all_cov[np.ix_([i],np.arange(0,rank2),np.arange(rank,rank+rank2**2))]=tmp_mixed.T
        

        var_rec_test[i]=var_train+np.einsum('ki,kk,ki->i',W_tot,all_cov[i],W_tot_var)
    data_rec_test=all_mean@W_tot+training_data_mean+err_train_mean#+data_U_test@W#+np.zeros_like(all_mean@W_tot)
    return data_rec_test, var_rec_test, err_train, var_train
