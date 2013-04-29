# References:
#
# http://software.intel.com/en-us/intel-mkl
# https://code.google.com/p/numexpr/wiki/NumexprVML
# http://scikit-learn.org/dev/auto_examples/applications/plot_hmm_stock_analysis.html
# http://scikit-learn.org/dev/auto_examples/plot_isotonic_regression.html#example-plot-isotonic-regression-py
# http://scikit-learn.org/dev/auto_examples/decomposition/plot_pca_3d.html#example-decomposition-plot-pca-3d-py

from __future__ import print_function
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os.path
import cPickle as pickle
from matplotlib import ticker

data_dir = './'
backends = ['anaconda', 'anaconda+mkl']


def plot_results(algo, datas, xlabel, ylabel, note, factor=None):
    plt.clf()
    fig1, ax1 = plt.subplots()
    plt.figtext(0.90, 0.94, "Note: " + note, va='top', ha='right')
    w, h = fig1.get_size_inches()
    fig1.set_size_inches(w*1.5, h)
    ax1.set_xscale('log')
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.get_xaxis().set_minor_locator(ticker.NullLocator())
    ax1.set_xticks(datas[0][:,0])
    ax1.grid(color="lightgrey", linestyle="--", linewidth=1, alpha=0.5)
    if factor:
        ax1.set_xticklabels([str(int(x)) for x in datas[0][:,0]/factor])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlim(datas[0][0,0]*.9, datas[0][-1,0]*1.1)
    plt.suptitle("%s Performance" % (algo), fontsize=28)

    for backend, data in zip(backends, datas):
        N = data[:, 0]
        plt.plot(N, data[:, 1], 'o-', linewidth=2, markersize=5, label=backend)
        plt.legend(loc='upper left', fontsize=18)

    plt.savefig(algo + '.png')

def load_data(algo):
    datas = []
    for backend in backends:
        filename = backend + '-' + algo + '.pkl'
        in_pickle = os.path.join(data_dir, filename)
        with open(in_pickle,'r') as data_file:
            data = pickle.load(data_file)
            datas.append(data)
    return datas

if __name__ == '__main__':
    plot_results('DGEMM',
                 load_data('DGEMM'),
                 r'Matrix Size',
                 'GFLOP/s',
                 'higher is better')

    plot_results('Cholesky',
                 load_data('Cholesky'),
                 r'Matrix Size',
                 'GFLOP/s',
                 'higher is better')

    plot_results('NumExpr',
                 load_data('NumExpr'),
                 r'Array Size',
                 'GB/s',
                 'higher is better')
