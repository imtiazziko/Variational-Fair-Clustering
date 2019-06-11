import numpy as np
from matplotlib import pyplot as plt
import os.path as osp

def plot_clusters_vs_lambda(X_org,l,filename,dataset,lmbda,balance,fairness_error):

        K = max(l) +1
        COLORSX = np.array(['rD','gP'])
        plt.figure(1,figsize=(6.4,4.8))
        plt.ion()
        plt.clf()
        
        group = ['cluster 1', 'cluster 2']
        for k in range(K):
            idx = np.asarray(np.where(l == k)).squeeze()
            plt.plot(X_org[idx,0],X_org[idx,1],COLORSX[k],label = group[k]);
        
        if dataset == 'Synthetic':        
            tmp_title = '$\lambda$ = {}, balance = {: .2f}'.format(lmbda,balance)
        else:
             tmp_title = '$\lambda$ = {}, fairness Error = {: .2f}'.format(lmbda,fairness_error)
        plt.title(tmp_title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
        plt.show()
        plt.close('all')

def plot_fairness_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, E_cluster_set):

    
        if not osp.exists(savefile):            
            np.savez(savefile, lmbdas = lmbdas, fairness_error = fairness_error_set, E_cluster = E_cluster_set)
        else:
            data = np.load(savefile)

        lmbdas = data['lmbdas']
        fairness_error_set = data['fairness_error']
        E_cluster_set = data['E_cluster']

        if cluster_option == 'kmeans':
            label_cluster = 'K-means'
        elif cluster_option == 'ncut':
            label_cluster = 'Ncut'
            
        dataset = (filename.split('_')[-1].split('.'))[0]
        
        title = '{} Dataset ---- Fair {}'.format(dataset,label_cluster)

        ylabel1 = 'Fairness error'
        ylabel2 = '{} energy'.format(label_cluster)


        length = 19
        plt.ion()
        fig, ax1 = plt.subplots()
        ax1.set_xlim ([0,length])
        ax2 = ax1.twinx()

        ax1.plot(lmbdas[:length], fairness_error_set[:length], '--rD' , linewidth=2.5, label = ylabel1)

        ax2.plot(lmbdas[:length], E_cluster_set[:length], '--bP' , linewidth=3, label = ylabel2)
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(ylabel1, color = 'r')
        ax2.set_ylabel(ylabel2,color = 'b')
        ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 0.8))
        ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.9))

        fig.suptitle(title)
        fig.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
        plt.show()
        plt.close('all')
        
def plot_convergence(cluster_option, filename, E_fair):
    
    # Plot original fair clustering energy
    
    if cluster_option == 'kmeans':
        label_cluster = 'K-means'
    elif cluster_option == 'ncut':
        label_cluster = 'Ncut'
        
    length = len(E_fair)
    iter_range  = list(range(1,length+1))
    plt.figure(1,figsize=(6.4,4.8))
    plt.ion()
    plt.clf()
    ylabel = 'Fair {} objective'.format(label_cluster)
    plt.plot(iter_range, E_fair, 'r-' , linewidth=2.2)
    plt.xlabel('outer iterations')
    plt.ylabel(ylabel)
    plt.xlim(1,length)
    plt.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
    plt.show()
    plt.close('all')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    