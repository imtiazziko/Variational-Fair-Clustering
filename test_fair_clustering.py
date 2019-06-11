from IPython import get_ipython
def __reset__(): get_ipython().magic('reset -sf')
import argparse
import os
import os.path as osp
import numpy as np
from src.fair_clustering import fair_clustering, km_init
import src.util as util
from src.dataset_load import read_dataset,dataset_names
from src.util  import get_fair_accuracy, get_fair_accuracy_proportional, normalizefea
from data_visualization import plot_clusters_vs_lambda, plot_fairness_vs_clusterE, plot_convergence
import random
import pdb



def  main(args):

    np.random.seed(1)
    random.seed(1)
    
    
    output_path = args.output_path
    data_dir = args.data_dir
    
    ## Select dataset
    dataset = args.dataset

    # ###  Clustering  options
    
    cluster_option = args.cluster_option
    
    ## plotting options
    plot_option_clusters_vs_lambda = args.plot_option_clusters_vs_lambda
    plot_option_fairness_vs_clusterE = args.plot_option_fairness_vs_clusterE
    plot_option_convergence = args.plot_option_convergence
    
    pdb.set_trace()
    # ###  Data load
    X_org, demograph, K = read_dataset(dataset)
    
    V_list =  [np.array(demograph == j) for j in np.unique(demograph)]
    V_sum =  [x.sum() for x in V_list]
#    J = len(V_sum)
    N,D = X_org.shape
    
    
    # demographic probability for each V_j
    
    u_V = [x/N for x in V_sum]  #proportional
    
    # Normalize Features
    
    X = normalizefea(X_org)
    
    #############################################################################
    
    ######################## Run Fair clustering #################################
    
    #############################################################################
    #    
    fairness = True # Setting False only runs unfair clustering
    
    elapsetimes = []
    balance_set = []
    fairness_error_set = []
    E_cluster_set = []
    bestacc = 1e10
    bestbalance = -1
    
    if args.lmbda is None:      
        lmbdas = np.arange(1,19,1).tolist()
    else:
        lmbdas = [args.lmbda]
        
    length_lmbdas = len(lmbdas)
    
    l = None
    
    affinity_path = osp.join(data_dir,dataset+'_affinity_ncut.npz')
    
    if cluster_option == 'ncut':
        knn = 20
        if not os.path.exists(affinity_path):
            A = util.create_affinity(X,knn,savepath = affinity_path)
        else:
            A = util.create_affinity(X,knn,W_path = affinity_path)
    
    
    init_C_path = osp.join(data_dir,dataset+'_init_{}_{}.npy'.format(cluster_option,K))
    
    for count,lmbda in enumerate(lmbdas):
        print('Inside Lambda ',lmbda)
    
        if not os.path.exists(init_C_path):
            
            C_init,_ = km_init(X,K,'kmeans')
            np.save(init_C_path,C_init)
            
        else:
            
            C_init = np.load(init_C_path) # Load initial seeds
    
            
        if cluster_option == 'ncut':
            
            C,l,elapsed,S,E = fair_clustering(X, K, u_V, V_list, lmbda, fairness, cluster_option, C_init, A = A)
            
        else:
            
            C,l,elapsed,S,E = fair_clustering(X, K, u_V, V_list, lmbda, fairness, cluster_option, C_init)
      
    
        balance, _ = get_fair_accuracy(u_V,V_list,l,N,K)
        fairness_error = get_fair_accuracy_proportional(u_V,V_list,l,N,K)
    
        print('lambda = {}, \n fairness_error {: .2f} and \n balance = {: .2f}'.format(lmbda, fairness_error, balance))
    
            
        # Plot the figure with clusters
        
        if dataset in ['Synthetic', 'Synthetic-unequal'] and plot_option_clusters_vs_lambda == True:
            
            filename = osp.join(output_path, 'cluster-plot_fair_{}-{}_lambda_{}.png'.format(cluster_option,dataset,lmbda))
            plot_clusters_vs_lambda(X_org,l,filename,dataset,lmbda,balance,fairness_error)
    #
        if balance>bestbalance:
           bestbalance = balance
           best_lambda_balance = lmbda

        if fairness_error<bestacc:
            bestacc = fairness_error
            best_lambda_acc = lmbda
            
            
        if plot_option_convergence == True and count == 1:
            
            filename = osp.join(output_path,'Fair_{}_convergence_{}.png'.format(cluster_option,dataset))
            E_fair = E['fair_cluster_E']
            plot_convergence(cluster_option, filename, E_fair)
    
    
    
        print('Best fairness_error %0.4f' %bestacc,'|Error lambda = ', best_lambda_acc)
        print('Best balance %0.4f' %bestbalance,'|Balance lambda = ', best_lambda_balance)
        elapsetimes.append(elapsed)
        balance_set.append(balance)
        fairness_error_set.append(fairness_error)
        E_cluster_set.append(E['cluster_E'][-1])
    
        
    avgelapsed = sum(elapsetimes)/len(elapsetimes)
    print ('avg elapsed ',avgelapsed)
    
    
    
    if plot_option_fairness_vs_clusterE == True and length_lmbdas > 1:
    
        
        savefile = osp.join(data_dir,'Fair_{}_fairness_vs_clusterE_{}.npz'.format(cluster_option,dataset))
        filename = osp.join(output_path,'Fair_{}_fairness_vs_clusterE_{}.png'.format(cluster_option,dataset))
        plot_fairness_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, E_cluster_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clustering with Fairness Constraints")
    # dataset
    parser.add_argument('-d', '--dataset', type=str, default='Synthetic-unequal',
                        choices=dataset_names())
    # clustering method
    parser.add_argument('--cluster_option', type=str, default='kmeans')
    
    # Plot options
    parser.add_argument('--plot_option_clusters_vs_lambda', action='store_true',
                        help="plot clusters in 2D w.r.t lambda")
    parser.add_argument('--plot_option_fairness_vs_clusterE', action='store_true',
                        help="plot clustering original energy w.r.t fairness")
    parser.add_argument('--plot_option_convergence', action='store_true',
                        help="plot convergence of the fair clustering energy")
    
    #Lambda
    parser.add_argument('--lmbda', type=float, default=None) # None run in a range of different lambdas
    
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--output_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'outputs'))
    main(parser.parse_args())
