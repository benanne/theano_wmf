import time

import numpy as np

import theano
import theano.tensor as T

import ops


def expressions_YTYpR_byY(Y, lambda_reg):
    f = Y.shape[1] - 1 # f = number of factors

    Y_e = T.set_subtensor(Y[:, f], T.ones((Y.shape[0],))) # factors with biases replaced by a column of ones.
    YTY = T.dot(Y_e.T, Y_e)

    R = T.eye(f + 1) # reguarization matrix
    R = T.set_subtensor(R[f, f], 0.0) # don't regularize the biases
    R *= lambda_reg

    YTYpR = YTY + R

    b_y = Y[:, f]
    byY = T.dot(b_y, Y_e)

    return YTYpR, byY


def batch_solve_expression(A_batch, B_batch):
    Binv_batch = ops.batched_inv(B_batch)
    R_batch = ops.batched_dot(A_batch.dimshuffle(0, 'x', 1), Binv_batch) # need to turn the A vectors into single-row matrices for this
    return R_batch[:, 0, :] # get rid of spurious dimension


def batch_update_expression(batch_index, Y, indptr, indices, data, YTYpR, byY, batch_size):
    m = indptr.shape[0] - 1 # m = number of users
    f = Y.shape[1] - 1 # f = number of factors

    lo = batch_index * batch_size
    hi = T.minimum((batch_index + 1) * batch_size, m)
    current_batch_size = hi - lo

    lo_batch = indptr[lo]
    hi_batch = indptr[hi] # hi - 1 + 1

    i_batch = indices[lo_batch:hi_batch]
    s_batch = data[lo_batch:hi_batch]
    Y_batch = Y[i_batch]

    b_y_batch = Y_batch[:, f]
    Y_e_batch = T.set_subtensor(Y_batch[:, f], T.ones((Y_batch.shape[0],)))

    # precompute the left hand side of the dot product for computing A for the entire batch.
    a_lhs_batch = (1 - b_y_batch) * s_batch + 1

    # also precompute the right hand side of the dot product for computing B for the entire batch.
    b_rhs_batch = Y_e_batch * s_batch.dimshuffle(0, 'x')

    # b_y = Y[:, f]
    # Y_e = T.set_subtensor(Y[:, f], T.ones((Y.shape[0],)))

    # scan iteration helper function that computes A and B for the current index
    def fn(k):
        # k_lo, k_hi = indptr[k], indptr[k + 1]
        # i_u = indices[k_lo:k_hi]
        # s_u = data[k_lo:k_hi]

        # Y_u = Y_e[i_u]
        # b_y_u = b_y[i_u]

        # A = T.dot((1 - b_y_u) * s_u + 1, Y_u)
        # B = T.dot(Y_u.T, Y_u * s_u.dimshuffle(0, 'x'))

        lo_iter = indptr[k] - lo_batch
        hi_iter = indptr[k + 1] - lo_batch

        s_u = s_batch[lo_iter:hi_iter]
        Y_u = Y_e_batch[lo_iter:hi_iter]
        a_lhs_u = a_lhs_batch[lo_iter:hi_iter]
        b_rhs_u = b_rhs_batch[lo_iter:hi_iter]

        A = T.dot(a_lhs_u, Y_u)
        B = T.dot(Y_u.T, b_rhs_u)

        return A, B

    (A_batch, B_batch), dummy_updates = theano.scan(fn, sequences=T.arange(lo, hi), name='AB_iter')

    A_batch -= byY.dimshuffle('x', 0)
    B_batch += YTYpR.dimshuffle('x', 0, 1)

    X_batch = batch_solve_expression(A_batch, B_batch)

    return X_batch


def factorize(S, num_factors, batch_size, lambda_reg=1e-5, num_iterations=20, init_std=0.01, verbose=False):
    """
    factorize a given sparse matrix using the Weighted Matrix Factorization algorithm by
    Hu, Koren and Volinsky.

    This is a GPU-only implementation in Theano. It does not run on the CPU because some
    of the custom ops use scikits.cuda.

    S: 'surplus' confidence matrix, i.e. C - I where C is the matrix with confidence weights.
        S is sparse while C is not (and the sparsity pattern of S is the same as that of
        the preference matrix, so this matrix doesn't need to be specified separately).

    num_factors: the number of factors.

    batch_size: size of the batches used for batched matrix inversion on the GPU. Make this
        big enough to benefit from the speedup, but don't make it too big or you will run
        out of memory.

    lambda_reg: the value of the regularization constant.

    num_iterations: the number of iterations to run the algorithm for. Each iteration consists
        of two steps, one to recompute U given V, and one to recompute V given U.

    init_std: the standard deviation of the Gaussian with which V is initialized.

    verbose: print a bunch of stuff during training, including timing information.

    returns:
        U, V: factor matrices. If bias=True, the last columns of the matrices contain the biases.
    """
    num_users, num_items = S.shape


    if verbose:
        print "precomputing transpose..."

    ST = S.T.tocsr()


    if verbose:
        print "copying data to GPU..."

    ## define shared variables for everything that persists across the computation of a single batch
    # input data
    indptr = theano.shared(S.indptr.astype('int32'))
    indices = theano.shared(S.indices.astype('int32'))
    data = theano.shared(S.data.astype(theano.config.floatX))

    indptr_t = theano.shared(ST.indptr.astype('int32'))
    indices_t = theano.shared(ST.indices.astype('int32'))
    data_t = theano.shared(ST.data.astype(theano.config.floatX))

    # output (factors)
    U = theano.shared(np.zeros((num_items, num_factors), dtype=theano.config.floatX)) # no need to initialize U randomly, it will be overwritten anyway
    V = theano.shared(np.random.randn(num_items, num_factors).astype(theano.config.floatX) * init_std)

    # things to precompute once per (half-)iteration (stays constant across batches)
    UTUpR = theano.shared(np.zeros((num_factors, num_factors), dtype=theano.config.floatX)) # this is precomputed each iteration
    buU = theano.shared(np.zeros((num_factors,), dtype=theano.config.floatX)) # this is precomputed each iteration

    VTVpR = theano.shared(np.zeros((num_factors, num_factors), dtype=theano.config.floatX)) # this is precomputed each iteration
    bvV = theano.shared(np.zeros((num_factors,), dtype=theano.config.floatX)) # this is precomputed each iteration


    ## symbolic batch index
    batch_index = T.iscalar('batch_index')


    if verbose:
        print "compiling functions..."

    ## functions for precomputation per half-iteration
    # for U (in terms of V)
    new_VTVpR, new_bvV = expressions_YTYpR_byY(V, lambda_reg)

    precompute_for_U = theano.function([], [], updates=[
        (VTVpR, new_VTVpR),
        (bvV, new_bvV),
    ])

    # for V (in terms of U)
    new_UTUpR, new_buU = expressions_YTYpR_byY(U, lambda_reg)

    precompute_for_V = theano.function([], [], updates=[
        (UTUpR, new_UTUpR),
        (buU, new_buU),
    ])


    ## functions for batch updates
    # for U (in terms of V)
    lo_U = batch_index * batch_size
    hi_U = T.minimum((batch_index + 1) * batch_size, num_users)

    batch_update_expr_U = batch_update_expression(batch_index, V, indptr, indices, data, VTVpR, bvV, batch_size)
    batch_update_U = theano.function([batch_index], [], updates=[
        (U, T.set_subtensor(U[lo_U:hi_U], batch_update_expr_U)),
    ])

    # for V (in terms of U)
    lo_V = batch_index * batch_size
    hi_V = T.minimum((batch_index + 1) * batch_size, num_items)

    batch_update_expr_V = batch_update_expression(batch_index, U, indptr_t, indices_t, data_t, UTUpR, buU, batch_size)
    batch_update_V = theano.function([batch_index], [], updates=[
        (V, T.set_subtensor(V[lo_V:hi_V], batch_update_expr_V)),
    ])


    ## functions that perform half-iterations
    num_batches_U = int(np.ceil(num_users / float(batch_size)))
    num_batches_V = int(np.ceil(num_items / float(batch_size)))

    def recompute_factors_U():
        precompute_for_U()
        for b in xrange(num_batches_U):
            print b # DEBUG TODO
            batch_update_U(b)

    def recompute_factors_V():
        precompute_for_V()
        for b in xrange(num_batches_V):
            print b # DEBUG TODO
            batch_update_V(b)


    if verbose:
        print "running ALS algorithm"
        start_time = time.time()


    for i in xrange(num_iterations):
        if verbose:
            print "iteration %d" % i
            print "  recompute user factors U"

        recompute_factors_U()

        if verbose:
            print "  time since start: %.3f seconds" % (time.time() - start_time)
            print "  recompute item factors V"

        recompute_factors_V()

        if verbose:
            print "  time since start: %.3f seconds" % (time.time() - start_time)

    return U.get_value(), V.get_value()



