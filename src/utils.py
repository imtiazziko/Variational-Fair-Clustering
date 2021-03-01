import numpy as np
import os
import sys
import errno
import shutil
import os.path as osp
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import scipy.sparse as sps
import timeit
from pyflann import FLANN
import multiprocessing

SHARED_VARS = {}
SHARED_array = {}

class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def normalizefea(X):
    """
    L2 normalize
    """
    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X_out = X/(feanorm[:,None]**0.5)
    return X_out

def get_V_jl(x,l,N,K):
    x = x.squeeze()
    temp =  np.zeros((N,K))
    index_cluster = l[x]
    temp[(x,index_cluster)]=1
    temp = temp.sum(0)
    return temp

def get_fair_accuracy(u_V,V_list,l,N,K):
    # pdb.set_trace()
    V_j_list  = np.array([get_V_jl(x,l,N,K) for x in V_list])
    
    balance = np.zeros(K)
    J = len(V_list)
    for k in range(K):
        V_j_list_k = V_j_list[:,k].copy()
        balance_temp = np.tile(V_j_list_k,[J,1])
        balance_temp = balance_temp.T/np.maximum(balance_temp,1e-20)
        mask = np.ones(balance_temp.shape, dtype=bool)
        np.fill_diagonal(mask,0)
        balance[k] = balance_temp[mask].min()
        
#    approx_j_per_K = N/(K*V_j_list.shape[0])
#    error = np.abs(V_j_list - approx_j_per_K)
#    error = error.sum()/N
    
    
    return balance.min(), balance.mean()

def get_fair_accuracy_proportional(u_V,V_list,l,N,K):

    V_j_list  = np.array([get_V_jl(x,l,N,K) for x in V_list])
    clustered_uV = V_j_list/sum(V_j_list)
#    balance = V_j_list/sum(V_j_list)
    fairness_error = np.zeros(K)
    u_V =np.array(u_V)
    
    for k in range(K):
        fairness_error[k] = (-u_V*np.log(np.maximum(clustered_uV[:,k],1e-20))+u_V*np.log(u_V)).sum()
    
    return fairness_error.sum()


def create_affinity(X, knn, scale = None, alg = "annoy", savepath = None, W_path = None):
    N,D = X.shape
    if W_path is not None:
        if W_path.endswith('.mat'):
            W = sio.loadmat(W_path)['W']
        elif W_path.endswith('.npz'):
            W = sparse.load_npz(W_path)
    else:
        
        print('Compute Affinity ')
        start_time = timeit.default_timer()
        if alg == "flann":
            print('with Flann')
            flann = FLANN()
            knnind,dist = flann.nn(X,X,knn, algorithm = "kdtree",target_precision = 0.9,cores = 5);
            # knnind = knnind[:,1:]
        else:
            nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
            dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N),knn-1)
        col = knnind[:,1:].flatten()
        if scale is None:
            data = np.ones(X.shape[0]*(knn-1))
        elif scale is True:
            scale = np.median(dist[:,1:])
            data = np.exp((-dist[:,1:]**2)/(2 * scale ** 2)).flatten() 
        else:
            data = np.exp((-dist[:,1:]**2)/(2 * scale ** 2)).flatten()

        W = sparse.csc_matrix((data, (row, col)), shape=(N,N),dtype=np.float)
        W = (W + W.transpose(copy=True)) /2
        elapsed = timeit.default_timer() - start_time
        print(elapsed)         

        if isinstance(savepath,str):
            if savepath.endswith('.npz'):
                sparse.save_npz(savepath,W)
            elif savepath.endswith('.mat'):
                sio.savemat(savepath,{'W':W})
            
    return W



### supporting functions to make parallel updates of clusters
    
def n2m(a):
   """
   Return a multiprocessing.Array COPY of a numpy.array, together
   with shape, typecode and matrix flag.
   """
   if not isinstance(a, np.ndarray): a = np.array(a)
   return multiprocessing.Array(a.dtype.char, a.flat, lock=False), tuple(a.shape), a.dtype.char, isinstance(a, np.matrix)

def m2n(buf, shape, typecode, ismatrix=False):
   """
   Return a numpy.array VIEW of a multiprocessing.Array given a
   handle to the array, the shape, the data typecode, and a boolean
   flag indicating whether the result should be cast as a matrix.
   """
   a = np.frombuffer(buf, dtype=typecode).reshape(shape)
   if ismatrix: a = np.asmatrix(a)
   return a

def mpassing(slices):

   i,k = slices
   Q_s,kernel_s_data,kernel_s_indices,kernel_s_indptr,kernel_s_shape = get_shared_arrays('Q_s','kernel_s_data','kernel_s_indices','kernel_s_indptr','kernel_s_shape')
   # kernel_s = sps.csc_matrix((SHARED_array['kernel_s_data'],SHARED_array['kernel_s_indices'],SHARED_array['kernel_s_indptr']), shape=SHARED_array['kernel_s_shape'], copy=False)
   kernel_s = sps.csc_matrix((kernel_s_data,kernel_s_indices,kernel_s_indptr), shape=kernel_s_shape, copy=False)
   Q_s[i,k] = kernel_s[i].dot(Q_s[:,k])
#    return Q_s


def new_shared_array(shape, typecode='d', ismatrix=False):
   """
   Allocate a new shared array and return all the details required
   to reinterpret it as a numpy array or matrix (same order of
   output arguments as n2m)
   """
   typecode = np.dtype(typecode).char
   return multiprocessing.Array(typecode, int(np.prod(shape)), lock=False), tuple(shape), typecode, ismatrix

def get_shared_arrays(*names):
   return [m2n(*SHARED_VARS[name]) for name in names]

def init(*pargs, **kwargs):
   SHARED_VARS.update(pargs, **kwargs)

####
