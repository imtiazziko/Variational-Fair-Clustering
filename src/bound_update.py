# -*- coding: utf-8 -*-

import numpy as np
import timeit
from src.progressBar import printProgressBar
from numba import  jit
import numexpr as ne

def normalize(S_in):

    maxcol = S_in.max(1)[:,np.newaxis]
    S_in = ne.evaluate('S_in - maxcol')
    S_out = np.exp(S_in)
    S_out_sum = S_out.sum(1)[:,np.newaxis]
    # S_out = np.divide(S_out,S_out_sum)
    S_out = ne.evaluate('S_out/S_out_sum')
    
    return S_out

def normalize_2(S_in):
    S_in_sum = S_in.sum(1)[:,np.newaxis]
    # S_in = np.divide(S_in,S_in_sum)
    S_in = ne.evaluate('S_in/S_in_sum')
    return S_in

def bound_energy(S, S_in, a_term, b_term, L, bound_lambda, batch = False):

    E = np.nansum((S*np.log(np.maximum(S,1e-15)) - S*np.log(np.maximum(S_in,1e-15)) + a_term*S + b_term*S))

    return E

@jit(parallel=True)
def compute_b_j_parallel(J,S,V_list,u_V):
    result = [compute_b_j(V_list[j],u_V[j],S) for j in range(J)]
    return result


def compute_b_j(V_j,u_j,S_):
    N,K = S_.shape
    V_j = V_j.astype('float')
    S_sum = S_.sum(0)
    R_j = ne.evaluate('u_j*(1/S_sum)')
    F_j_a = np.tile((ne.evaluate('u_j*V_j')),[K,1]).T
    F_j_b = np.maximum(np.tile(np.dot(V_j,S_),[N,1]),1e-15)
    F_j = ne.evaluate('R_j - (F_j_a/F_j_b)')

    return F_j


@jit
def get_S_discrete(l,N,K):
    x = range(N)
    temp =  np.zeros((N,K),dtype=float)
    temp[(x,l)]=1
    return temp

            
def bound_update(a_p, u_V, V_list, bound_lambda, L, bound_iteration = 200, debug=False):
    
    """
    """
    start_time = timeit.default_timer()
    print("Inside Bound Update . . .")
    N,K = a_p.shape
    oldE = float('inf')
    J = len(u_V)
    

# Initialize the S
    S = np.exp((-a_p))
    S = normalize_2(S)

    for i in range(bound_iteration):
        printProgressBar(i + 1, bound_iteration,length=12)
        # S = np.maximum(S, 1e-20)
        S_in = S.copy()
        
        # Get a and b 
        terms = - a_p.copy()

        b_j_list = compute_b_j_parallel(J,S,V_list,u_V)
        b_j_list = sum(b_j_list)
        b_term = ne.evaluate('bound_lambda * b_j_list')
        terms = ne.evaluate('(terms - b_term)/L')
        S_in_2 = normalize(terms)  
        S = ne.evaluate('S_in * S_in_2')
        S = normalize_2(S)
        if debug:
            print('b_term = {}'.format(b_term[0:10]))
            print('a_p = {}'.format(a_p[0:10]))
            print('terms = {}'.format(terms[0:10]))
            print('S = {}'.format(S[0:10]))
            #Check for trivial solutions
            l = np.argmax(S,axis=1)
            if len(np.unique(l))<S.shape[1]:
                S = S_in.copy()

        E = bound_energy(S, S_in, a_p, b_term, L, bound_lambda)
        # print('Bound Energy {} at iteration {} '.format(E,i))
        report_E = E
        
        if (i>1 and (abs(E-oldE)<= 1e-5*abs(oldE))):
            print('Converged')
            break

        else:
            oldE = E; report_E = E

    elapsed = timeit.default_timer() - start_time
    print('\n Elapsed Time in bound_update', elapsed)
    l = np.argmax(S,axis=1)
    
    return l,S,report_E

