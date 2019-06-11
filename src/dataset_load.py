#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:28:15 2019

@author: iziko
"""
import numpy as np
import os
from sklearn.datasets import make_blobs
import sys
import requests, zipfile, io
import pandas
from matplotlib import pyplot as plt
#from glob import glob
#import re
from sklearn.preprocessing import scale
#import os.path as osp



__datasets = ['Adult', 'Bank', 'Synthetic', 'Synthetic-unequal']

def dataset_names():

    return __datasets


def read_dataset(name):


    if name not in __datasets:
        raise KeyError("Dataset not implemented:",name)
        
    elif name == 'Synthetic':
        
        n_samples = 400

        centers = [(1, 1), (2.1, 1), (1, 5), (2.1, 5)]
        data, sex_num = make_blobs(n_samples=n_samples, n_features=2, cluster_std=0.1,
                  centers=centers, shuffle=False, random_state=1)
        
        index = n_samples//2
        sex_num[0:index] = 0
        sex_num[index:n_samples] = 1
        K = 2
        
    elif name == 'Synthetic-unequal':
        
        n_samples = 400

        sample_list = [150,150,50,50]
        centers = [(1, 1), (2.1, 1), (1, 3.5), (2.1, 3.5)]
        data, sex_num = make_blobs(n_samples=sample_list, n_features=2, cluster_std=0.13,
                  centers=centers, shuffle=False, random_state=1)
        
        index = sample_list[0]+sample_list[1]
        sex_num[0:index] = 0
        sex_num[index:] = 1
        K = 2
        
    elif name == 'Adult':
        
        data_path = './data/adult.data'
        race_is_sensitive_attribute = 0
        
        if race_is_sensitive_attribute==1:
            m = 5
        else:
            m = 2
        n = 20000
        K = 20
        if (not os.path.exists(data_path)): 
            print('Adult data set does not exist in current folder --- Have to download it')
            r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', allow_redirects=True)
            if r.status_code == requests.codes.ok:
                print('Download successful')
            else:
                print('Could not download Adult data set - please download it manually')
                sys.exit()
            open(data_path, 'wb').write(r.content)
        
        df = pandas.read_csv(data_path, sep=',',header=None)
        df = df[:n]
        
        sens_attr = 9
        sex = df[sens_attr]
        sens_attributes = list(set(sex.astype(str).values))   # =[' Male', ' Female']
        df = df.drop(columns=[sens_attr])
        sex_num = np.zeros(n, dtype=int)
        sex_num[sex.astype(str).values == sens_attributes[1]] = 1


        #dropping non-numerical features and normalizing data
        cont_types = np.where(df.dtypes=='int')[0]   # =[0,2,4,9,10,11]
        df = df.iloc[:,cont_types]
        data = np.array(df.values, dtype=float)
        
        data = data[:,[0,1,2,5]]
        
        #Scale data
        data = scale(data, axis = 0)

    elif name == 'Bank':
        
        n= 4000
        K = 2
        data_path = './data/bank-additional-full.csv'

        if (not os.path.exists(data_path)): 

            print('Bank dataset does not exist in current folder --- Have to download it')
            r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip', allow_redirects=True)
            if r.status_code == requests.codes.ok:
                print('Download successful')
            else:
                print('Could not download - please download')
                sys.exit()

            z = zipfile.ZipFile(io.BytesIO(r.content))
            # z.extract('bank-additional/bank-additional-full.csv','./data')
            open(data_path, 'wb').write(z.read('bank-additional/bank-additional-full.csv'))

        df = pandas.read_csv(data_path,sep=';')
        print(df.columns)
#        shape = df.shape
        
#        df = df.loc[np.random.choice(df.index, n, replace=False)]
        sex = df['marital'].astype(str).values
        sens_attributes = list(set(sex))
        sens_attributes.remove('unknown')
        df1 = df.loc[df['marital'] == sens_attributes[0]][:n]
        df2 = df.loc[df['marital'] == sens_attributes[1]][:n]
        df3 = df.loc[df['marital'] == sens_attributes[2]][:n]
        
        df = [df1, df2, df3]
        df = pandas.concat(df)
        
        sex = df['marital'].astype(str).values
        
        df = df[['age','duration','euribor3m', 'nr.employed']].values 

        sens_attributes = list(set(sex))
        sex_num = np.zeros(df.shape[0], dtype=int)
        sex_num[sex == sens_attributes[1]] = 1
        sex_num[sex == sens_attributes[2]] = 2

        data = np.array(df, dtype=float)
        
        #Scale data
        data = scale(data, axis = 0)
#        
#        data = data[,:]
    else:
        pass

    return data, sex_num, K
    
    
if __name__=='__main__':
    
#    dataset = 'Synthetic'
    dataset = 'Synthetic-unequal'
    
    data, sex_num, K = read_dataset(dataset)
    filename = '{}.png'.format(dataset)
    
    COLORSX = np.array(['bv','kP'])
    
    plt.figure(1,figsize=(6.4,4.8))
    plt.clf()
    
#     marker_edge_color = ['r',]
    group = ['demographic 1 ({})', 'demographic 2 ({})']
    for k in range(K):
        idx = np.asarray(np.where(sex_num == k)).squeeze()
        label_text = group[k].format(len(idx))
        plt.plot(data[idx,0],data[idx,1],COLORSX[k],label = label_text);
    plt.title(dataset)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
    plt.show()
    plt.close('all')
    
    
#    COLORSX = np.array(['bv','kP'])
#    plt.figure(1,figsize=(6.4,4.8))
#    plt.clf()
#    
##     marker_edge_color = ['r',]
#    group = ['demographic 1 ({})', 'demographic 2 ({})']
#    for k in range(K):
#        idx = np.asarray(np.where(sex_num == k)).squeeze()
#        label_text = group[k].format(len(idx))
#        plt.plot(data[idx,0],data[idx,1],COLORSX[k],label = label_text);
##	     plt.plot(C[k,0],C[k,1],COLORSX[5]);
#    plt.title('Synthetic (equal)')
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
#    plt.show()
#    plt.close('all')
    
    
    
    
    
    
    
    
    