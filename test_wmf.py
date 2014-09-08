import numpy as np
import theano_wmf

np.random.seed(123)


B = np.load("/home/sander/git/wmf/test_matrix.pkl")

# convert into a surplus confidence matrix

alpha = 2.0
epsilon = 1e-6

S = B.copy()
S.data = alpha * np.log(1 + S.data / epsilon)


num_factors = 40 + 1
num_iterations = 1
batch_size = 10000
lambda_reg = 1e-5


U, V = theano_wmf.factorize(S, num_factors, batch_size, lambda_reg, num_iterations, init_std=0.01, verbose=True)
