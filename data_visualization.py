import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import os.path as osp

def plot_clusters_vs_lambda(X_org,l,filename,dataset, lmbda, fairness_error):

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
            tmp_title = '$\lambda$ = {}, fairness Error = {: .2f}'.format(lmbda,fairness_error)
        else:
             tmp_title = '$\lambda$ = {}, fairness Error = {: .2f}'.format(lmbda,fairness_error)
        plt.title(tmp_title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
        plt.show()
        plt.close('all')

def plot_fairness_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, min_balance_set, avg_balance_set, E_cluster_set, save = True):

        if not osp.exists(savefile) or save == True:
            np.savez(savefile, lmbdas = lmbdas, min_balance_set = min_balance_set, avg_balance_set = avg_balance_set, fairness_error = fairness_error_set, E_cluster = E_cluster_set)
        else:
            data = np.load(savefile)
            lmbdas = data['lmbdas']
            fairness_error_set = data['fairness_error']
            E_cluster_set = data['E_cluster']
        # pdb.set_trace()
        if cluster_option == 'kmeans':
            label_cluster = 'K-means'
        elif cluster_option == 'ncut':
            label_cluster = 'Ncut'
        elif cluster_option == 'kmedian':
            label_cluster = 'K-medians'
            
        dataset = (filename.split('_')[-1].split('.'))[0]
        
        title = '{} Dataset ---- Fair {}'.format(dataset,label_cluster)

        ylabel1 = 'Fairness error'
        ylabel2 = '{} discrete energy'.format(label_cluster)


        length = len(lmbdas)
        plt.ion()
        plt.rcParams.update({'font.size': 13})
        fig, ax1 = plt.subplots(dpi=80)
        # ax1.set_xlim ([0,length])
        ax2 = ax1.twinx()
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.plot(lmbdas[:length], fairness_error_set[:length], '--rD' , linewidth=2.5, label = ylabel1)

        ax2.plot(lmbdas[:length], E_cluster_set[:length], '--bP' , linewidth=3, label = ylabel2)
        ax1.set_xlabel(r'$\lambda$', fontsize=15)
        ax1.set_ylabel(ylabel1, color = 'r', fontsize=15)
        ax2.set_ylabel(ylabel2,color = 'b', fontsize=15)
        # ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 0.6))
        ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.7))

        fig.suptitle(title, fontsize=16)
        fig.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
        plt.show()
        plt.close('all')

def plot_K_vs_clusterE(cluster_option, savefile, filename, K_list, E_cluster_set, E_0_cluster_set, save = True):

        if not osp.exists(savefile) or save == True:
            np.savez(savefile, K_list = K_list, E_cluster_set = E_cluster_set, E_0_cluster_set = E_0_cluster_set)
        else:
            data = np.load(savefile)
            K_list = data['K_list']
            E_cluster_set = data['E_cluster_set']
            E_0_cluster_set = data['E_0_cluster_set']
        # pdb.set_trace()
        if cluster_option == 'kmeans':
            label_cluster = 'K-means'
        elif cluster_option == 'ncut':
            label_cluster = 'Ncut'
        elif cluster_option == 'kmedian':
            label_cluster = 'K-medians'

        dataset = (savefile.split('_')[-1].split('.'))[0]

        title = '{} Dataset ---- Fair {}'.format(dataset,label_cluster)

        ylabel = '{} discrete energy'.format(label_cluster)

        legend_1 = 'Variational Fair {}'.format(label_cluster)
        legend_2 = 'Vanilla {}'.format(label_cluster)

        plt.figure(1,figsize=(7,5), dpi=80)
        plt.rcParams.update({'font.size': 15})
        plt.ion()
        plt.clf()
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.plot(K_list, E_cluster_set, '--gD' , linewidth=2.2)
        plt.plot(K_list, E_0_cluster_set, '--bP' , linewidth=2.2)
        plt.xlabel('Number of clusters (K)')
        plt.ylabel(ylabel)
        plt.legend([legend_1, legend_2], loc='upper center')
        plt.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
        plt.show()
        plt.close('all')


def plot_balance_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, min_balance_set, avg_balance_set, E_cluster_set, save = True):


        if not osp.exists(savefile) or save == True:
            np.savez(savefile, lmbdas = lmbdas, fairness_error = fairness_error_set, min_balance_set = min_balance_set, avg_balance_set = avg_balance_set, E_cluster = E_cluster_set)
        else:
            data = np.load(savefile)
            lmbdas = data['lmbdas']
            avg_balance_set = data['avg_balance_set']
            E_cluster_set = data['E_cluster']
        # pdb.set_trace()
        if cluster_option == 'kmeans':
            label_cluster = 'K-means'
        elif cluster_option == 'ncut':
            label_cluster = 'Ncut'
        elif cluster_option == 'kmedian':
            label_cluster = 'K-medians'

        dataset = (filename.split('_')[-1].split('.'))[0]

        title = '{} Dataset ---- Fair {}'.format(dataset,label_cluster)

        ylabel1 = ' Average Balance'
        ylabel2 = '{} discrete energy'.format(label_cluster)


        length = len(lmbdas)
        plt.ion()
        fig, ax1 = plt.subplots()
        # ax1.set_xlim ([0,length])
        ax2 = ax1.twinx()

        ax1.plot(lmbdas[:length], avg_balance_set[:length], '--rD' , linewidth=2.5, label = ylabel1)

        ax2.plot(lmbdas[:length], E_cluster_set[:length], '--bP' , linewidth=3, label = ylabel2)
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(ylabel1, color = 'r')
        ax2.set_ylabel(ylabel2,color = 'b')
        ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 0.6))
        ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.7))

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
    elif cluster_option == 'kmedian':
        label_cluster = 'K-medians'
        
    length = len(E_fair)
    iter_range  = list(range(1,length+1))
    plt.figure(1,figsize=(7,5), dpi=80)
    plt.rcParams.update({'font.size': 15})
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
    
if __name__ == '__main__':
    cluster_option = 'ncut'
    data_dir = 'data/Bank'
    dataset = 'Bank'
    output_path = 'outputs'
    # savefile = osp.join(data_dir,'Fair_{}_K_vs_clusterEdiscrete_{}.npz'.format(cluster_option,dataset))
    # filename = osp.join(data_dir,'Fair_{}_K_vs_clusterEdiscrete_{}_3.png'.format(cluster_option,dataset))
    # plot_K_vs_clusterE(cluster_option, savefile, filename, [], [], [], save=False)
    #
    savefile = osp.join(data_dir,'Fair_{}_fairness_vs_clusterEdiscrete_{}.npz'.format(cluster_option,dataset))
    filename = osp.join(data_dir,'Fair_{}_fairness_vs_clusterEdiscrete2_{}.png'.format(cluster_option,dataset))
    plot_fairness_vs_clusterE(cluster_option, savefile, filename, [], [], [], [], [], save = False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    