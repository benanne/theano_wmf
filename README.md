theano_wmf
==========

Weighted matrix factorization on the GPU with Theano and scikits.cuda

This is a work in progress and probably has a lot of bugs. Needs to be tested against another implementation for correctness.

At the moment I'm stuck with this. It does not seem to be that much faster than a CPU version. For small num_factors, the 'scan' part is the bottleneck, and I don't see how this can be parallelized further. For large num_factors, the 'BatchedInvOp' is the bottleneck. This could maybe be made a bit faster by making a destructive version (that destroys the input), but it probably won't make a big difference.