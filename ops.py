# TODO: implement destructive version of BatchedInvOp
# TODO: implement optimization to replace batched_inv{destructive=False} with batched_inv{destructive=True} if applicable.



import numpy as np 
import theano
import theano.tensor as T

import theano.sandbox.cuda as cuda
from theano.misc.pycuda_utils import to_gpuarray

import scikits.cuda
from scikits.cuda import linalg
from scikits.cuda import cublas

import pycuda.gpuarray

import theano.misc.pycuda_init

import string

linalg.init()



class ScikitsCudaOp(cuda.GpuOp): # base class for shared code between scikits.cuda-based ops
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def output_type(self, inp):
        raise NotImplementedError

    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)()])


def bptrs(a):
    """
    Pointer array when input represents a batch of matrices.

    taken from scikits.cuda tests/test_cublas.py
    """
    
    return pycuda.gpuarray.arange(a.ptr,a.ptr+a.shape[0]*a.strides[0],a.strides[0],
                dtype=cublas.ctypes.c_void_p)


def gpu_dot_batched(bx_gpu, by_gpu, bc_gpu, transa='N', transb='N', handle=None):
    """
    uses cublasSgemmBatched to compute a bunch of dot products in parallel
    """
    if handle is None:
        handle = scikits.cuda.misc._global_cublas_handle

    assert len(bx_gpu.shape) == 3
    assert len(by_gpu.shape) == 3
    assert len(bc_gpu.shape) == 3
    assert bx_gpu.dtype == np.float32
    assert by_gpu.dtype == np.float32 
    assert bc_gpu.dtype == np.float32

    # Get the shapes of the arguments
    bx_shape = bx_gpu.shape
    by_shape = by_gpu.shape
    
    # Perform matrix multiplication for 2D arrays:
    alpha = np.float32(1.0)
    beta = np.float32(0.0)
    
    transa = string.lower(transa)
    transb = string.lower(transb)

    if transb in ['t', 'c']:
        N, m, k = by_shape
    elif transb in ['n']:
        N, k, m = by_shape
    else:
        raise ValueError('invalid value for transb')

    if transa in ['t', 'c']:
        N2, l, n = bx_shape
    elif transa in ['n']:
        N2, n, l = bx_shape
    else:
        raise ValueError('invalid value for transa')

    if l != k:
        raise ValueError('objects are not aligned')

    if N != N2:
        raise ValueError('batch sizes are not the same')

    if transb == 'n':
        lda = max(1, m)
    else:
        lda = max(1, k)

    if transa == 'n':
        ldb = max(1, k)
    else:
        ldb = max(1, n)

    ldc = max(1, m)

    # construct pointer arrays needed for cublasCgemmBatched
    bx_arr = bptrs(bx_gpu)
    by_arr = bptrs(by_gpu)
    bc_arr = bptrs(bc_gpu)

    cublas.cublasSgemmBatched(handle, transb, transa, m, n, k, alpha, by_arr.gpudata,
                lda, bx_arr.gpudata, ldb, beta, bc_arr.gpudata, ldc, N)


class BatchedDotOp(ScikitsCudaOp):
    def make_node(self, inp1, inp2):
        inp1 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp1))
        inp2 = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 3 # (batch, a, b)
        assert inp2.ndim == 3

        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            bx = inputs[0]
            by = inputs[1]

            input_shape_x = bx[0].shape # (batch, a, b)
            input_shape_y = by[0].shape # (batch, b, c)

            output_shape = (input_shape_x[0], input_shape_x[1], input_shape_y[2]) # (batch, a, c)

            bz = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if bz[0] is None or bz[0].shape != output_shape:
                bz[0] = cuda.CudaNdarray.zeros(output_shape)

            input_bx_pycuda = to_gpuarray(bx[0])
            input_by_pycuda = to_gpuarray(by[0])
            output_b_pycuda = to_gpuarray(bz[0])

            # fancy native batched version
            gpu_dot_batched(input_bx_pycuda, input_by_pycuda, output_b_pycuda)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


batched_dot = BatchedDotOp()



class BatchedInvOp(ScikitsCudaOp):
    def __init__(self, destructive=False):
        super(BatchedInvOp, self).__init__()
        assert destructive == False # TODO: destructive op not supported for now (need to add destroy_map and optimization)
        self.destructive = destructive

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.destructive == other.destructive)

    def __hash__(self):
        return (hash(type(self)) ^
            hash(self.destructive))

    def __str__(self):
        return "%s{destructive=%s}" % (self.__class__.__name__, self.destructive)

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        # reusable allocations
        pivot_alloc = [None]
        info_alloc = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            size = input_shape[1] # matrices to invert are (size x size)
            batch_size = input_shape[0]

            z = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if z[0] is None or z[0].shape != input_shape:
                z[0] = cuda.CudaNdarray.zeros(input_shape)
                pivot_alloc[0] = pycuda.gpuarray.empty((batch_size, size), np.int32)
                info_alloc[0] = pycuda.gpuarray.zeros(batch_size, np.int32)

            input_pycuda = to_gpuarray(inputs[0][0])
            output_pycuda = to_gpuarray(z[0])
            pivot = pivot_alloc[0]
            info = info_alloc[0]

            # construct pointer arrays for batched operations
            input_arr = bptrs(input_pycuda)
            output_arr = bptrs(output_pycuda)

            if not self.destructive:
                input_pycuda = input_pycuda.copy() # to prevent destruction of the input

            handle = scikits.cuda.misc._global_cublas_handle

            # perform LU factorization
            cublas.cublasSgetrfBatched(handle, size, input_arr.gpudata, size, pivot.gpudata, info.gpudata, batch_size)
            # the LU factorization is now in input_pycuda (destructive operation!)

            # use factorization to perform inversion
            cublas.cublasSgetriBatched(handle, size, input_arr.gpudata, size, pivot.gpudata, output_arr.gpudata, size, info.gpudata, batch_size)
            # the inverted matrices are now in output_pycuda

                    
        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


batched_inv = BatchedInvOp()
# batched_inv_destructive = BatchedInvOp(destroy_input=True)