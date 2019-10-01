#!/usr/bin/python 
# -*- coding: utf-8 -*-
'''
Comparison with ICML and other methods

'''

import numpy as np
import scipy.io as sio
import os.path as osp
from scipy import sparse
from src.fair_clustering import NormalizedCutEnergy_discrete, NormalizedCutEnergy, compute_energy_fair_clustering
from src.bound_update import  get_S_discrete

from src.util import get_fair_accuracy, get_fair_accuracy_proportional, normalizefea
import warnings
warnings.filterwarnings('ignore')

def fairness_term_V_j(u_j,S,V_j):
    V_j = V_j.astype('float')
    S_term = np.dot(V_j,S)
    S_term = u_j*np.log(np.maximum(S_term,1e-20))
    term = u_j*np.log(np.maximum(S.sum(0),1e-20)) - S_term

    return term


def convert_to_mat(filename):
    datas = np.load(filename+'.npz')
    X_org = datas['X_org']
    demograph = datas['demograph']
    K = datas['K']
    sio.savemat(filename+'.mat',{'X_org':X_org,'demograph':demograph,'K':K})

def convert_to_affinity(filename):
    A = sparse.load_npz(filename+'.npz')
    sio.savemat(filename+'.mat',{'A':A})

data_dir = './data'
dataset = 'Adult'
# affinity_path = osp.join(data_dir,dataset+'_affinity_ncut_final')
# convert_to_affinity(affinity_path)
# data_path = osp.join(data_dir,dataset)
# convert_to_mat(data_path)

data_path = osp.join(data_dir,'icml_normalized_'+dataset+'.mat')

data = sio.loadmat(data_path)
#
l = data ['l'] - 1
l = l.squeeze()
# data = sio.loadmat('data/Synthetic_icml_compare.mat')
data_path = osp.join(data_dir,dataset+'.mat')
data =sio.loadmat(data_path)
demograph = data['demograph']
# X = data['X']
X_org = data['X_org']
X = normalizefea(X_org)
K = data['K'][0][0]
# K = 10
# K = 30
N  = X.shape[0]
V_list =  [np.array(demograph == j) for j in np.unique(demograph)]
V_sum =  [x.sum() for x in V_list]
J = len(V_sum)
u_V = [x/N for x in V_sum]
# N,D = X_org.shape
# J = len(u_V)
# # S = []
# # C = []
# # balance and Fairness error
balance,_ = get_fair_accuracy(u_V,V_list,l,N,K)
fairness_error = get_fair_accuracy_proportional(u_V,V_list,l,N,K)
# #
# #
method_cl = 'ncut'
S = get_S_discrete(l,N,K)
filename = osp.join(data_dir,dataset+'_affinity_ncut_final.mat')
# filename = osp.join(data_dir,dataset+'_affinity_ncut.mat')
A = sio.loadmat(filename)['A']
# bound_lambda = 1
currentE, clusterE, fairE, clusterE_discrete = compute_energy_fair_clustering(X, [], l, S, u_V, V_list,0, A = A, method_cl=method_cl)
#

## Ours
cluster_option = 'kmeans'
data_dir = 'data'
dataset = 'Adult'
output_path = 'outputs'
savefile = osp.join(data_dir,'Fair_{}_fairness_vs_clusterEdiscrete_{}.npz'.format(cluster_option,dataset))
# plot_fairness_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, E_cluster_set)
data = np.load(savefile)
lmbdas = data['lmbdas']
fairness_error_set = data['fairness_error']
E_cluster_set = data['E_cluster']
print(min(E_cluster_set[1:]))
avg_balance_set = data['avg_balance_set']
min_balance_set = data['min_balance_set']
print('done')


#if __name__ == '__main__':
#    main()

    
    
