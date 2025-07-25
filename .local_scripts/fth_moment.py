import cupy as cp

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
      moments (cp.ndarray): An (n x 4) array of moments.
    
    Returns:
      cov (cp.ndarray): A (n^2 x n^2) covariance matrix.
    """
    n = moments.shape[0]
    
    # Precompute pairwise expectations E[Z_i Z_j]
    EZ = cp.empty((n, n))
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


    cov_single=moments[:,1]-cp.diag(moments[:,0]**2)

    # Build the covariance matrix for the vectorized outer product (of size n^2)
    cov = cp.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            row = i * n + j
            for k in range(n):
                for l in range(n):
                    col = k * n + l
                    exp4 = fourth_moment(i, j, k, l)
                    cov[row, col] = exp4 - EZ[i, j] * EZ[k, l]

    cov_mixed=cp.zeros((n*n, n))
    for i in range(n):
        for j in range(n):
            row = i * n + j
            for k in range(n):
                exp3 = third_moment(i, j, k)
                cov_mixed[row, k] = exp3- EZ[i, j] * moments[k,0]



    return EZ.reshape(-1), cov, cov_mixed

# Example: Suppose we have 3 variables with the following moments:
# For each variable i, moments[i] = [mean, second moment, third moment, fourth moment]
moments = cp.array([
    [1.0, 2.0, 3.0, 5.0],  # moments for Z_1
    [0.5, 1.2, 1.8, 2.5],  # moments for Z_2
    [0.0, 1.0, 0.0, 2.0]   # moments for Z_3
])

covariance_matrix = compute_outer_product_covariance(moments)
print("Covariance matrix of the outer product Z ⊗ Z (vectorized):")
print(covariance_matrix)
