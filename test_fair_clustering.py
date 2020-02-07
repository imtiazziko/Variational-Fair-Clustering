import argparse
import os
import sys
import os.path as osp
import numpy as np
from src.fair_clustering import fair_clustering, km_init
import src.util as util
from src.dataset_load import read_dataset,dataset_names
from src.util  import get_fair_accuracy, get_fair_accuracy_proportional, normalizefea, Logger, str2bool
from data_visualization import plot_clusters_vs_lambda, plot_fairness_vs_clusterE, plot_convergence, plot_balance_vs_clusterE
import random
import warnings
warnings.filterwarnings('ignore')
import pdb

def  main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    ## Options
    dataset = args.dataset
    cluster_option = args.cluster_option
    data_dir = osp.join(args.data_dir, dataset)
    output_path = data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ## plotting options
    plot_option_clusters_vs_lambda = args.plot_option_clusters_vs_lambda
    plot_option_fairness_vs_clusterE = args.plot_option_fairness_vs_clusterE
    plot_option_balance_vs_clusterE = args.plot_option_balance_vs_clusterE
    plot_option_convergence = args.plot_option_convergence
    
        # ###  Data load
    savepath_compare =  osp.join(data_dir,dataset+'.npz')
    if not os.path.exists(savepath_compare):
        X_org, demograph, K = read_dataset(dataset, data_dir)
        if X_org.shape[0]>200000:
            np.savez_compressed(savepath_compare, X_org = X_org, demograph = demograph, K = K)
        else:
            np.savez(savepath_compare, X_org=X_org, demograph=demograph, K=K)

    else:
        datas = np.load(savepath_compare)
        X_org = datas['X_org']
        demograph = datas['demograph']
        K = datas['K'].item()

    log_path = osp.join(data_dir,cluster_option+'_log.txt')
    sys.stdout = Logger(log_path)
    # Normalize Features
    X = normalizefea(X_org)
    N, D = X.shape
    print('Cluster number for dataset {} is {}'.format(dataset,K))
    V_list =  [np.array(demograph == j) for j in np.unique(demograph)]
    V_sum =  [x.sum() for x in V_list]
    print('Balance of the dataset {}'.format(min(V_sum)/max(V_sum)))
#    J = len(V_sum)
    
    
    # demographic probability for each V_j
    
    u_V = [x/N for x in V_sum]  #proportional
    
    #############################################################################
    
    ######################## Run Fair clustering #################################
    
    #############################################################################
    #    
    fairness = True # Setting False only runs unfair clustering
    
    elapsetimes = []
    avg_balance_set = []
    min_balance_set = []
    fairness_error_set = []
    E_cluster_set = []
    E_cluster_discrete_set = []
    bestacc = 1e10
    best_avg_balance = -1
    best_min_balance = -1
    
    if args.lmbda is None:
        lmbdas = np.arange(10,50,2).tolist()
    else:
        lmbdas = [args.lmbda]
        
    length_lmbdas = len(lmbdas)
    
    l = None


    if (not 'A' in locals()) and cluster_option == 'ncut':
        alg_option = 'flann' if N>50000 else 'None'
        affinity_path = osp.join(data_dir, cluster_option +'_affinity_ncut.npz')
        knn = 20
        if not os.path.exists(affinity_path):
            A = util.create_affinity(X,knn,savepath = affinity_path, alg=alg_option)
        else:
            A = util.create_affinity(X,knn,W_path = affinity_path)

    
    init_C_path = osp.join(data_dir,'_init_{}_{}.npy'.format(cluster_option,K))
    
    for count,lmbda in enumerate(lmbdas):
        print('Inside Lambda ',lmbda)
    
        if not os.path.exists(init_C_path):
            print('Generating initial seeds')
            C_init,_ = km_init(X,K,'kmeans_plus')
            np.save(init_C_path,C_init)
            
        else:
            
            C_init = np.load(init_C_path) # Load initial seeds
    
            
        if cluster_option == 'ncut':
            
            C,l,elapsed,S,E = fair_clustering(X, K, u_V, V_list, lmbda, fairness, cluster_option, C_init, A = A)
            
        else:
            
            C,l,elapsed,S,E = fair_clustering(X, K, u_V, V_list, lmbda, fairness, cluster_option, C_init)
      
        min_balance, avg_balance = get_fair_accuracy(u_V,V_list,l,N,K)
        fairness_error = get_fair_accuracy_proportional(u_V,V_list,l,N,K)
    
        print('lambda = {}, \n fairness_error {: .2f} and \n avg_balance = {: .2f} \n min_balance = {: .2f}'.format(lmbda, fairness_error, avg_balance, min_balance))
    
            
        # Plot the figure with clusters
        
        if dataset in ['Synthetic', 'Synthetic-unequal'] and plot_option_clusters_vs_lambda == True:
            
            filename = osp.join(output_path, 'cluster_output', 'cluster-plot_fair_{}-{}_lambda_{}.png'.format(cluster_option,dataset,lmbda))
            plot_clusters_vs_lambda(X_org,l,filename,dataset,lmbda, min_balance_set, avg_balance_set,fairness_error)
    #
        if avg_balance>best_avg_balance:
           best_avg_balance = avg_balance
           best_lambda_avg_balance = lmbda

        if min_balance>best_min_balance:
           best_min_balance = min_balance
           best_lambda_min_balance = lmbda

        if fairness_error<bestacc:
            bestacc = fairness_error
            best_lambda_acc = lmbda
            
            
        if plot_option_convergence == True:
            
            filename = osp.join(output_path,'Fair_{}_convergence_{}.png'.format(cluster_option,dataset))
            E_fair = E['fair_cluster_E']
            plot_convergence(cluster_option, filename, E_fair)
    
    
    
        print('Best fairness_error %0.4f' %bestacc,'|Error lambda = ', best_lambda_acc)
        print('Best  Avg balance %0.4f' %best_avg_balance,'| Avg Balance lambda = ', best_lambda_avg_balance)
        print('Best  Min balance %0.4f' %best_min_balance,'| Min Balance lambda = ', best_lambda_min_balance)
        elapsetimes.append(elapsed)
        avg_balance_set.append(avg_balance)
        min_balance_set.append(min_balance)
        fairness_error_set.append(fairness_error)
        E_cluster_set.append(E['cluster_E'][-1])
        E_cluster_discrete_set.append(E['cluster_E_discrete'][-1])
    
        
    avgelapsed = sum(elapsetimes)/len(elapsetimes)
    print ('avg elapsed ',avgelapsed)
    
    
    if plot_option_fairness_vs_clusterE == True and length_lmbdas > 1:
    

        savefile = osp.join(data_dir,'Fair_{}_fairness_vs_clusterEdiscrete_{}.npz'.format(cluster_option,dataset))
        filename = osp.join(output_path,'Fair_{}_fairness_vs_clusterEdiscrete_{}.png'.format(cluster_option,dataset))
        plot_fairness_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, min_balance_set, avg_balance_set, E_cluster_discrete_set)

    if plot_option_balance_vs_clusterE == True and length_lmbdas > 1:

        savefile = osp.join(data_dir,'Fair_{}_balance_vs_clusterEdiscrete_{}.npz'.format(cluster_option,dataset))
        filename = osp.join(output_path,'Fair_{}_balance_vs_clusterEdiscrete_{}.png'.format(cluster_option,dataset))

        plot_balance_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, min_balance_set, avg_balance_set, E_cluster_discrete_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clustering with Fairness Constraints")
    parser.add_argument('--seed', type=int, default=None)  # None run in a range of different lambdas
    # dataset
    parser.add_argument('-d', '--dataset', type=str, default='Synthetic',
                        choices=dataset_names())
    # clustering method
    parser.add_argument('--cluster_option', type=str, default='ncut')
    
    # Plot options
    parser.add_argument('--plot_option_clusters_vs_lambda', default=False, type=str2bool,
                        help="plot clusters in 2D w.r.t lambda")
    parser.add_argument('--plot_option_fairness_vs_clusterE', default=False, type=str2bool,
                        help="plot clustering original energy w.r.t fairness")
    parser.add_argument('--plot_option_balance_vs_clusterE', default=False, type=str2bool,
                        help="plot clustering original energy w.r.t balance")
    parser.add_argument('--plot_option_convergence', default=False, type=str2bool,
                        help="plot convergence of the fair clustering energy")
    
    #Lambda
    parser.add_argument('--lmbda', type=float, default=50) # None run in a range of different lambdas

    
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--output_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'outputs'))
    main(parser.parse_args())
