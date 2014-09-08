import numpy as np

import theano
import theano.tensor as T

a = np.random.randn(200, 40).astype('float32')

A = theano.shared(a)

start_idx = T.iscalar()
end_idx = T.iscalar()

updates = [(A, T.set_subtensor(A[start_idx:end_idx], A[start_idx:end_idx] - A.sum(0)))]
f = theano.function([start_idx, end_idx], [], updates=updates)

theano.printing.debugprint(f)