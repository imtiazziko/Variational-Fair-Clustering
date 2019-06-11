from IPython import get_ipython
def __reset__(): get_ipython().magic('reset -sf')

import os
import os.path as osp
import numpy as np
from src.fair_clustering import fair_clustering, km_init
import src.util as util
from src.dataset_load import read_dataset
from src.util  import get_fair_accuracy, get_fair_accuracy_proportional, normalizefea
from matplotlib import pyplot as plt
import matplotlib as mp
from data_visualization import plot_clusters_vs_lambda
import random
#import pdb


np.random.seed(1)
random.seed(1)


output_path = './outputs'
data_dir = './data'

## Select dataset
#dataset = 'synthetic'
#dataset = 'Bank'
dataset = 'Adult'

# ###  Clustering  options

cluster_option = 'kmeans'
#cluster_option = 'ncut'

## plotting options
#plot_option_clusters_vs_lambda = True
#plot_option_clusterenergy_vs_error = False
#plot_convergence = False
output_path = './outputs'
data_dir = './data'


savefile = 'Fair_{}_lambda_vs_kl_clustering_{}.npz'.format(cluster_option,dataset)


data = np.load(savefile)

lmbdas = data['lmbadas'][0:19]
fairness_error_set = data['kl_error'][0:19]
E_cluster_set = data['E_cluster'][0:19]


savefile = osp.join(data_dir,'Fair_{}_fairness_vs_clusterE_{}.npz'.format(cluster_option,dataset))

np.savez(savefile, lmbdas = lmbdas, fairness_error = fairness_error_set, E_cluster = E_cluster_set)

# #pdb.set_trace()

#data = np.load(savefile)
#lmbdas = data['lmbadas']
#fairness_error_set = data['fairness_error']
#E_cluster_set = data['E_cluster']
#
#if cluster_option == 'kmeans':
#    label_cluster = 'K-means'
#elif cluster_option == 'ncut':
#    label_cluster = 'Ncut'
#
#title = '{} Dataset ---- Fair {}'.format(dataset,label_cluster)
#
#ylabel1 = 'Fairness error'
##ylabel2 = 'K-means energy'
#ylabel2 = '{} energy'.format(label_cluster)
#
##filename = 'Fair_kmeans_lambda_vs_kl_clustering_synthetic.png'
#filename = 'Fair_{}_lambda_vs_kl_clustering_{}_K_{}_2.png'.format(cluster_option,dataset,K)
#
#
##length = 19
#fig, ax1 = plt.subplots()
#ax1.set_xlim ([0,length])
#ax2 = ax1.twinx()
#
#ax1.plot(lmbdas[:length], fairness_error_set[:length], '--rP' , linewidth=2.5, label = ylabel1)
#
#ax2.plot(lmbdas[:length], E_cluster_set[:length], '--bP' , linewidth=3, label = ylabel2)
##    mp.rcParams.update({'font.size': 13})
#ax1.set_xlabel(r'$\lambda$')
#ax1.set_ylabel(ylabel1, color = 'r')
#ax2.set_ylabel(ylabel2,color = 'b')
##ax2.set_ylim([18.5,19])
#ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 0.8))
#ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.9))
#
#fig.suptitle(title)
#fig.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
#plt.show()
#plt.close('all')
