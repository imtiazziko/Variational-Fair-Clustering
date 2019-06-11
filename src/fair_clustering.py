#!/usr/bin/python 
# -*- coding: utf-8 -*-


import numpy as np
from scipy import sparse
import math
from sklearn.metrics.pairwise import euclidean_distances as ecdist
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _init_centroids
#import multiprocessing
from src.bound_update import bound_update,normalize_2, get_S_discrete
from src.util  import get_fair_accuracy, get_fair_accuracy_proportional
import timeit





def kmeans_update(X,tmp):
    """
    
    """
    c1 = X[tmp,:].mean(axis = 0)
    
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
    
def KernelBound(A, K, S, current_clustering):
    N = current_clustering.size
    unaries = np.zeros((N, K), dtype=np.float)
    d = A.sum(axis=1)
    for i in range(K):

        S_i = S[:,i]
        volume_s_i = np.dot(np.transpose(d), S_i)
        volume_s_i = volume_s_i[0,0]
        #print volume_s_i
        temp = np.dot(np.transpose(S_i), A.dot(S_i)) / volume_s_i / volume_s_i
        temp = temp * d
        #print temp.shape
        temp2 = temp + np.reshape( - 2 * A.dot(S_i) / volume_s_i, (N,1))
        #print type(temp2)
        unaries[:,i] = temp2.flatten()
        
    return unaries


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
    S_term = np.dot(V_j,S)
    S_term = u_j*np.log(np.maximum(S_term,1e-20))
    term = u_j*np.log(np.maximum(S.sum(0),1e-20)) - S_term
    
    return term


def compute_energy_fair_clustering(X, C, l, S, u_V, V_list, bound_lambda, A = None, method_cl='kmeans'):
    """
    compute fair clustering energy
    
    """
    J = len(u_V)
    e_dist = ecdist(X,C,squared =True)
    N,K = S.shape
    if method_cl =='kmeans':
        # K-means energy
        clustering_E = (S*e_dist).sum()
    elif method_cl =='ncut':
        
        clustering_E = NormalizedCutEnergy(A,S,l)
    
    # Fairness term 
    fairness_E = [fairness_term_V_j(u_V[j],S,V_list[j]) for j in range(J)]
    fairness_E = (bound_lambda*sum(fairness_E)).sum()
    
    E = clustering_E + fairness_E
    

    return E, clustering_E, fairness_E
    
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

    trivial_status = False
    S = []
    E_org = []
    E_cluster = []
    E_fair = []
    fairness_error = 0.0
    balance  = 0.0
    oldE = 1e100
    
    for i in range(100):
        oldC = C.copy()
        oldl = l.copy()
        oldS = S.copy()
        
        if i == 0:
            
            sqdist = ecdist(X,C,squared=True)
            S = normalize_2(np.exp((-sqdist)))
            a_p = S*sqdist
#                pass
            if method == 'ncut':
                S = get_S_discrete(l,N,K)
                sqdist = KernelBound(A, K, S, l)
                a_p = sqdist.copy()
            
        elif method == 'kmeans':
            
            # TODO: Make it parallel for each k
            
            print ('Inside k-means update')
            
            for k in range(C.shape[0]):
                tmp=np.asarray(np.where(l== k))
                if tmp.size !=1:
                    tmp = tmp.squeeze()
                else:
                    tmp = tmp[0]
                C[[k],] = kmeans_update(X,tmp)
                
            sqdist = ecdist(X,C,squared=True)
            a_p = S*sqdist
            
        elif method == 'ncut':
            print ('Inside ncut update')
            
            sqdist = KernelBound(A, K, S, l)
            a_p = sqdist.copy()
    
            
        if fairness ==True and lmbda!=0.0:
            
            if method == 'kmeans':
                l_check = km_le(X,C,None,None)
            else:
                l_check = a_p.argmin(axis=1)
            
            # Check for empty cluster
            if (len(np.unique(l_check))!=K):
                l,C,S,mode_index,trivial_status = restore_nonempty_cluster(X,K,oldl,oldC,oldS,ts)
                ts = ts+1
                if trivial_status:
                    break
                
            bound_iterations = 600
            
            fairness_error = get_fair_accuracy_proportional(u_V,V_list,l_check,N,K)
            print('fairness_error = {:0.4f}'.format(fairness_error))
                
            l,S,bound_E = bound_update(a_p,X, l, u_V, V_list, lmbda, bound_iterations)
            
            fairness_error = get_fair_accuracy_proportional(u_V,V_list,l,N,K)
            print('fairness_error = {:0.4f}'.format(fairness_error))
        
            
            
        else:
                
            if method == 'kmeans':
                S = normalize_2(np.exp((-a_p)))
                l = km_le(X,C,None,None)
            
            else:
                l = a_p.argmin(axis=1)
                S = get_S_discrete(l,N,K)
#                S = normalize_2(np.exp((-a_p)))
            
            
        if method == 'ncut' and lmbda==0:
            S = normalize_2(np.exp((-a_p)))
            
        currentE, clusterE, fairE = compute_energy_fair_clustering(X, C, l, S, u_V, V_list,lmbda, A = A, method_cl=method)    
        E_org.append(currentE)
        E_cluster.append(clusterE)
        E_fair.append(fairE)
        
        
        if (len(np.unique(l))!=K) or math.isnan(fairness_error):
            l,C,S,mode_index,trivial_status = restore_nonempty_cluster(X,K,oldl,oldC,oldS,ts)
            ts = ts+1
            if trivial_status:
                break

        if (i>1 and (abs(currentE-oldE)<= 1e-4*abs(oldE)) or balance>0.99):
            print('......Job  done......')
            break
            
        
        else:       
            oldE = currentE.copy()
    
    
    elapsed = timeit.default_timer() - start_time
    print(elapsed) 
    E = {'fair_cluster_E':E_org[1:],'fair_E':E_fair[1:],'cluster_E':E_cluster[1:]}
    
    return C,l,elapsed,S,E

#if __name__ == '__main__':
#    main()

    
    
