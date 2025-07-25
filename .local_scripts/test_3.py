import numpy as np
from scipy.optimize import shgo, Bounds, minimize

A=np.random.rand(360,100000).astype(np.float32)
U,S,V=np.linalg.svd(A,full_matrices=False)

U=U[:,:100].astype(np.float32)

def obj(V):
    return np.linalg.norm(U.reshape(-1)-V)

def cos1(V):
    return np.linalg.norm(V.reshape(U.shape),axis=1)-10


def cos2(V):
    V_tmp=V.reshape(U.shape)
    return np.linalg.norm(1/(V_tmp.shape[0]-1)*V_tmp.T@V_tmp-np.eye(100))

V0=U.reshape(-1)

bounds=[(-1,1), ]*V0.shape[0]

cons=({"type": "eq", "fun": cos1},
      {"type": "eq", "fun": cos2})

res=minimize(obj,x0=V0,bounds=bounds,constraints=cons, method="trust-constr")
