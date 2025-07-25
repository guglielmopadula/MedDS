import numpy as np

# Example arrays
a = np.arange(3)
b = np.arange(3)

# Create the NxN matrix where A[i, j, 0] = a[i] and A[i, j, 1] = b[j]
A = np.zeros((len(a), len(b), 2))

# Populate A[i, j, 0] = a[i] and A[i, j, 1] = b[j]
A[:, :, 0] = a[:, np.newaxis]  # broadcasting a[i]
A[:, :, 1] = b[np.newaxis, :]  # broadcasting b[j]


A=A.reshape(-1,2)

B = np.zeros((len(A), len(A), 4))

B[:,:,:2]=A[:,np.newaxis]
B[:,:,2:]=A[np.newaxis:]

B=B.astype(np.int64)

B=B.reshape(3,3,3,3,4)

sigma=np.random.rand(3,3)

B0=B[:,:,:,:,0]
B1=B[:,:,:,:,1]

print(sigma[B0,B1].shape)
