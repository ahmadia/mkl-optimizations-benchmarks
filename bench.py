# References:
#
# http://software.intel.com/en-us/intel-mkl
# https://code.google.com/p/numexpr/wiki/NumexprVML

from __future__ import print_function
import datetime
import sys
from scipy import stats
import numpy as np
import numexpr as ne
import time
import gc
import os.path
import cPickle as pickle

data_dir = './'

def time_dgemm(N=100, trials=3, dtype=np.double):
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    B = np.asarray(np.random.rand(N, N), dtype=dtype)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    for i in xrange(trials):
        C = A.dot(B)
    toc = time.time()-tic
    if gcold:
        gc.enable()
    return toc/trials, 2*N*N*N*1e-9

def time_cholesky(N=100, trials=3, dtype=np.double):
    A = np.asarray(np.random.rand(N, N), dtype=dtype)
    A = A*A.transpose() + N*np.eye(N)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    for i in xrange(trials):
        L = np.linalg.cholesky(A)
    toc = time.time()-tic
    if gcold:
        gc.enable()
    return toc/trials, N*N*N/3.0*1e-9

def time_numexpr(N=100, trials=3, dtype=np.double):
    x = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
    y = np.asarray(np.linspace(-1, 1, N), dtype=dtype)
    z = np.empty_like(x)
    gcold = gc.isenabled()
    gc.disable()
    tic = time.time()
    for i in xrange(trials):
        ne.evaluate('2*y+4*x', out = z)
    toc = time.time()-tic
    if gcold:
        gc.enable()
    return (toc/trials, dtype().itemsize*3*N*1e-9)

def test_timers():
    N = 500
    trials = 3
    dtype = np.double
    s, gflop = time_dgemm(N, trials, dtype)
    print("DGEMM   : N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gflop = time_cholesky(N, trials, dtype)
    print("Cholesky: N: %d s: %e GFLOP/s: %e" % (N, s, gflop/s))
    s, gbyte = time_numexpr(50000, trials, dtype)
    print("NumExpr : N: %d s: %e GBytes/s: %e" % (N, s, gbyte/s))

def bench(test_fun, Ns, trials, dtype=None):
    data = np.empty((len(Ns),2))
    print("%d tests" % len(Ns))
    tic = time.time()
    for i in xrange(len(Ns)):
        sys.stdout.write('.')
        sys.stdout.flush()
        if dtype is not None:
            out_tuple= test_fun(Ns[i],trials,dtype)
        else:
            out_tuple= test_fun(Ns[i],trials)

        if len(out_tuple) > 1:
            data[i,:] = (Ns[i], out_tuple[1]/out_tuple[0])
        else:
            data[i,:] = (Ns[i], out_tuple[0])
    print('done')
    toc = time.time() - tic
    print('tests took: %e seconds' % toc)
    return data

def dump_data(data, data_dir, backend, algo):
    filename = backend + '-' + algo + '.pkl'
    out_pickle = os.path.join(data_dir, filename)
    with open(out_pickle,'w') as data_file:
        pickle.dump(data, data_file)

if __name__ == '__main__':
    try:
        import mkl
        have_mkl = True
        backend = 'anaconda+mkl'
        print("Running with MKL Acceleration")
    except ImportError:
        have_mkl = False
        backend = 'anaconda'
        print("Running with normal backends")

    print("checking timers...")
    test_timers()
    logNs = np.arange(6,13.5,0.5) # uncomment to run the big stuff
#    logNs = np.arange(3,7,0.5) # uncomment to run quick tests
    Ns = np.exp2(logNs)
    trials = 5
    dtype = np.double

    print('benchmarking DGEMM')
    dgemm_data = bench(time_dgemm, Ns, trials, dtype)
    dump_data(dgemm_data, data_dir, backend, 'DGEMM')

    print('benchmarking Cholesky')
    cholesky_data = bench(time_cholesky, Ns, trials, dtype)
    dump_data(cholesky_data, data_dir, backend, 'Cholesky')

    print('benchmarking NumExpr')
    logNs = np.arange(12, 18.5, 0.5) # uncomment to run big tests
#    logNs = np.arange(6,13.5,0.5) # uncomment to run quick tests
    Ns = np.exp2(logNs)
    numexpr_data = bench(time_numexpr, Ns, trials, dtype)
    dump_data(numexpr_data, data_dir, backend, 'NumExpr')
