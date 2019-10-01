#!/usr/bin/python 
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import sparse
import math
from sklearn.metrics.pairwise import euclidean_distances as ecdist
from sklearn.metrics import pairwise_distances_chunked as pdist_chunk
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _init_centroids
#import multiprocessing

from src.bound_update import bound_update,normalize_2, get_S_discrete
from src.util  import get_fair_accuracy, get_fair_accuracy_proportional
import timeit
import src.util as util
import multiprocessing
import ray
from numba import  jit, prange
import numexpr as ne

@ray.remote
def kmeans_update(X,tmp):
    """

    """
    c1 = X[tmp,:].mean(axis = 0)

    return c1

# @ray.remote
# def kmeans_update(X,S,k):
#     """
#
#     """
#     # pdb.set_trace()
#     S_k = S[:,k]
#     c1 = np.dot(X.T,S_k)/S_k.sum()
#     return c1

@jit
def reduce_func(D_chunk, start):
    J = np.mean(D_chunk,axis=1)
    return J


def kmedian_update(tmp):
    """

    """
    # print("ID of process running worker: {}".format(os.getpid()))
    X = util.SHARED_VARS['X_s']
    X_tmp = X[tmp,:]
    D = pdist_chunk(X_tmp,reduce_func=reduce_func)
    J = next(D)
    j = np.argmin(J)
    c1 = X_tmp[j,:]
    return c1

def NormalizedCutEnergy(A, S, clustering):
    if isinstance(A, np.ndarray):
        d = np.sum(A, axis=1)

    elif isinstance(A, sparse.csc_matrix):
        
        d = A.sum(axis=1)

    maxclusterid = np.max(clustering)
    #print "max cluster id is: ", maxclusterid
    nassoc_e = 0;
    num_cluster = 0;
    for k in range(maxclusterid+1):
        S_k = S[:,k]
        #print S_k
        if 0 == np.sum(clustering==k):
             continue # skip empty cluster
        num_cluster = num_cluster + 1
        if isinstance(A, np.ndarray):
            nassoc_e = nassoc_e + np.dot( np.dot(np.transpose(S_k),  A) , S_k) / np.dot(np.transpose(d), S_k)
        elif isinstance(A, sparse.csc_matrix):
            nassoc_e = nassoc_e + np.dot(np.transpose(S_k), A.dot(S_k)) / np.dot(np.transpose(d), S_k)
            nassoc_e = nassoc_e[0,0]
    #print "number of clusters: ", num_cluster
    ncut_e = num_cluster - nassoc_e

    return ncut_e

def NormalizedCutEnergy_discrete(A, clustering):
    if isinstance(A, np.ndarray):
        d = np.sum(A, axis=1)

    elif isinstance(A, sparse.csc_matrix):

        d = A.sum(axis=1)

    maxclusterid = np.max(clustering)
    #print "max cluster id is: ", maxclusterid
    nassoc_e = 0;
    num_cluster = 0;
    for k in range(maxclusterid+1):
        S_k = np.array(clustering == k,dtype=np.float)
        #print S_k
        if 0 == np.sum(clustering==k):
             continue # skip empty cluster
        num_cluster = num_cluster + 1
        if isinstance(A, np.ndarray):
            nassoc_e = nassoc_e + np.dot( np.dot(np.transpose(S_k),  A) , S_k) / np.dot(np.transpose(d), S_k)
        elif isinstance(A, sparse.csc_matrix):
            nassoc_e = nassoc_e + np.dot(np.transpose(S_k), A.dot(S_k)) / np.dot(np.transpose(d), S_k)
            nassoc_e = nassoc_e[0,0]
    #print "number of clusters: ", num_cluster
    ncut_e = num_cluster - nassoc_e

    return ncut_e

@ray.remote
def KernelBound_k(A, d, S, N, k):
    S_i = S[:,k]
    volume_s_i = np.dot(np.transpose(d), S_i)
    volume_s_i = volume_s_i[0,0]
    #print volume_s_i
    temp = np.dot(np.transpose(S_i), A.dot(S_i)) / volume_s_i / volume_s_i
    temp = temp * d
    #print temp.shape
    temp2 = temp + np.reshape( - 2 * A.dot(S_i) / volume_s_i, (N,1))
    #print type(temp2)

    return temp2.flatten()

@jit
def km_le(X,M,method,sigma):
    
    """
    Discretize the assignments based on center
    
    """
    
    e_dist = ecdist(X,M)          
    l = e_dist.argmin(axis=1)
        
    return l

# Fairness term calculation
def fairness_term_V_j(u_j,S,V_j):
    V_j = V_j.astype('float')
    S_term = np.maximum(np.dot(V_j,S),1e-20)
    S_sum = np.maximum(S.sum(0),1e-20)
    S_term = ne.evaluate('u_j*(log(S_sum) - log(S_term))')
    return S_term


@ray.remote
def km_discrete_energy(e_dist,l,k):
    tmp = np.asarray(np.where(l== k)).squeeze()
    return np.sum(e_dist[tmp,k])

def compute_energy_fair_clustering(X, C, l, S, u_V, V_list, bound_lambda, A = None, method_cl='kmeans'):
    """
    compute fair clustering energy
    
    """
    print('compute energy')
    J = len(u_V)

    N,K = S.shape
    clustering_E_discrete = []
    if method_cl =='kmeans':
        e_dist = ecdist(X,C,squared =True)
        clustering_E = ne.evaluate('S*e_dist').sum()
        clustering_E_discrete = [km_discrete_energy.remote(e_dist,l,k) for k in range(K)]
        clustering_E_discrete = sum(ray.get(clustering_E_discrete))

    elif method_cl =='ncut':
        
        clustering_E = NormalizedCutEnergy(A,S,l)
        clustering_E_discrete = NormalizedCutEnergy_discrete(A,l)

    elif method_cl =='kmedian':
        e_dist = ecdist(X,C)
        clustering_E = ne.evaluate('S*e_dist').sum()
        clustering_E_discrete = [km_discrete_energy.remote(e_dist,l,k) for k in range(K)]
        clustering_E_discrete = sum(ray.get(clustering_E_discrete))
    
    # Fairness term 
    fairness_E = [fairness_term_V_j(u_V[j],S,V_list[j]) for j in range(J)]
    fairness_E = (bound_lambda*sum(fairness_E)).sum()
    
    E = clustering_E + fairness_E
    print('fair clustering energy = {}'.format(E))

    return E, clustering_E, fairness_E, clustering_E_discrete
    
def km_init(X,K,C_init):
    
    """
    Initial seeds
    """
    
    N,D = X.shape
    if isinstance(C_init,str):

        if C_init == 'kmeans_plus':
            M =_init_centroids(X,K,init='k-means++')
            l = km_le(X,M,None,None)
            
        elif C_init =='kmeans':
            kmeans = KMeans(n_clusters=K).fit(X)
            l =kmeans.labels_
            M = kmeans.cluster_centers_
    else:
        M = C_init.copy(); 
        l = km_le(X,M,None,None)
        
    del C_init

    return M,l

def restore_nonempty_cluster (X,K,oldl,oldC,oldS,ts):
        ts_limit = 2
        C_init = 'kmeans'
        if ts>ts_limit:
            print('not having some labels')
            trivial_status = True
            l =oldl.copy();
            C =oldC.copy();
            S = oldS.copy()

        else:
            print('try with new seeds')
            
            C,l =  km_init(X,K,C_init)
            sqdist = ecdist(X,C,squared=True)
            S = normalize_2(np.exp((-sqdist)))
            trivial_status = False
        
        return l,C,S,trivial_status


def fair_clustering(X, K, u_V, V_list, lmbda, fairness = False, method = 'kmeans', C_init = "kmeans_plus", A = None):
    
    """ 
    
    Proposed farness clustering method
    
    """
    N,D = X.shape
    start_time = timeit.default_timer()
    
    C,l =  km_init(X,K,C_init)
    assert len(np.unique(l)) == K
    ts = 0

    trivial_status = False # for empty cluster status
    S = []
    E_org = []
    E_cluster = []
    E_fair = []
    E_cluster_discrete = []
    fairness_error = 0.0
    balance  = 0.0
    oldE = 1e100

    maxiter = 100
    X_s = util.init(X_s =X)
    pool = multiprocessing.Pool(processes=10)
    if A is not None:
        A_s = ray.put(A)
        d =  A.sum(axis=1)
        d_s = ray.put(d)


    for i in range(maxiter):
        oldC = C.copy()
        oldl = l.copy()
        oldS = S.copy()
        
        if i == 0:
            if method == 'kmeans':
                sqdist = ecdist(X,C,squared=True)
                a_p = sqdist.copy()

            if method == 'kmedian':
                sqdist = ecdist(X,C)
                a_p = sqdist.copy()
            if method == 'ncut':
                S = get_S_discrete(l,N,K)
                result_id = []
                for i in range(K):
                    result_id.append(KernelBound_k.remote(A_s, d_s, S, N, i))
                sqdist = ray.get(result_id)
                sqdist = np.asarray(np.vstack(sqdist).T)
                a_p = sqdist.copy()

            
        elif method == 'kmeans':
            
            print ('Inside k-means update')
            result_ids = []
            for k in range(K):
                tmp=np.asarray(np.where(l== k))
                if tmp.size !=1:
                    tmp = tmp.squeeze()
                else:
                    tmp = tmp[0]

                result_ids.append(kmeans_update.remote(X,tmp))

            # # print(C)
            C = ray.get(result_ids)
            C = np.asarray(np.vstack(C))
            sqdist = ecdist(X,C,squared=True)
            a_p = sqdist.copy()
        elif method == 'kmedian':

            print ('Inside k-median update')
            tmp_list = [np.where(l==k)[0] for k in range(K)]
            result_ids = pool.map(kmedian_update,tmp_list)
            C = np.asarray(np.vstack(result_ids))
            sqdist = ecdist(X,C)
            a_p = sqdist.copy()

        elif method == 'ncut':
            print ('Inside ncut update')
            S = get_S_discrete(l,N,K)
            result_id = []
            for i in range(K):
                result_id.append(KernelBound_k.remote(A_s, d_s, S, N, i))
            sqdist = ray.get(result_id)
            sqdist = np.asarray(np.vstack(sqdist).T)
            a_p = sqdist.copy()

    
            
        if fairness ==True and lmbda!=0.0:

            l_check = a_p.argmin(axis=1)
            
            # Check for empty cluster
            if (len(np.unique(l_check))!=K):
                l,C,S,trivial_status = restore_nonempty_cluster(X,K,oldl,oldC,oldS,ts)
                ts = ts+1
                if trivial_status:
                    break
                
            bound_iterations = 1000

            l,S,bound_E = bound_update(a_p, u_V, V_list, lmbda, bound_iterations)
            
            fairness_error = get_fair_accuracy_proportional(u_V,V_list,l,N,K)
            print('fairness_error = {:0.4f}'.format(fairness_error))
        
            
            
        else:
                
            if method == 'ncut':
                l = a_p.argmin(axis=1)
                S = get_S_discrete(l,N,K)
            
            else:
                S = get_S_discrete(l,N,K)
                l = km_le(X,C,None,None)
            
        currentE, clusterE, fairE, clusterE_discrete = compute_energy_fair_clustering(X, C, l, S, u_V, V_list,lmbda, A = A, method_cl=method)
        E_org.append(currentE)
        E_cluster.append(clusterE)
        E_fair.append(fairE)
        E_cluster_discrete.append(clusterE_discrete)
        
        
        if (len(np.unique(l))!=K) or math.isnan(fairness_error):
            l,C,S,trivial_status = restore_nonempty_cluster(X,K,oldl,oldC,oldS,ts)
            ts = ts+1
            if trivial_status:
                break

        if (i>1 and (abs(currentE-oldE)<= 1e-3*abs(oldE))):
            print('......Job  done......')
            break
            
        
        else:       
            oldE = currentE.copy()

    pool.close()
    pool.join()
    pool.terminate()
    elapsed = timeit.default_timer() - start_time
    print(elapsed)
    E = {'fair_cluster_E':E_org,'fair_E':E_fair,'cluster_E':E_cluster, 'cluster_E_discrete':E_cluster_discrete}
    return C,l,elapsed,S,E

#if __name__ == '__main__':
#    main()

    
    
