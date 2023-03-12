# -*- coding: utf-8 -*-
"""
Beyesian Optimization
Created on Fri Dec  2 22:54:44 2022
@author: Zhaonan Meng, Ditra Matin

This .py file implements basic bayesian optimization for hyperparameters
"""
import os
import sys
import csv
import numpy as np
from scipy.stats import norm

# sklearn
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# optimization toolkit
dir_path = os.path.dirname(os.path.realpath(__file__))  # getting current directory
sys.path.append(dir_path)
from optimization import MBFGS

# Load input data
def load_data(path):
    try:
        with open(path,"r") as csvfile:
            arr = []
            csvreader = csv.reader(csvfile,delimiter=",")        
            for row in csvreader:
                arr.append(row)
        return np.array(arr)
    except:
        print("File not found!")
        return None
    
# Preprocess data set
def data_preprocessing(data):
    data = np.float64(data)         # convert string to floating point number
    train = data[:,0:-1]            # the first 13 columns: training(validation) data
    label = data[:,-1].astype(int)  # the last column: label -1/+1
    return train,label
 
# svc training (for cross_validation())   
def svc_train(train,label,C,gamma):
    '''
    Parameters                       
    ----------
    train : array
        Training data set.
    label : array
        Label (+1/-1).
    C : floating point number
        Regularization parameter.
    Gamma : floating point number
        Kernel scaling parameter.

    Returns
    -------
    clf: svc object
        Classifier
    '''
    # Support Vector Machine algorithms are not scale invariant, 
    # so it is highly recommended to scale your data. 
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel='rbf',C=C,gamma=gamma))
    clf.fit(train,label)
    return clf

# cross_validation 
def cross_validation(*args):
    '''
    Parameters  args = [C,gamma,k,train,label]
    ----------
    train : array                C : floating point number      
        Trainning data set.          Regularization parameter.
    label : array                Gamma : floating point number
        Label (+1/-1).               Kernel scaling parameter.  
    k : integer
        k-fold

    Returns
    -------
    performance: floating point number
        The final performance of the model
    '''
    if len(args) < 2:
        print("Too few arguments! Please at least specify C and gamma!")
        return None
    if len(args) == 2:
        # if there are 2 arguments, C and gamma are specified by args. k is 5 by default
        C = args[0]
        gamma = args[1]
        k = 5
        global train,label    # we use the training dataset outside the function
    else:
        C = args[0]
        gamma = args[1]
        k = args[2]
        if len(args) == 5:
            train = args[3]
            label = args[4]
    
    N = train.shape[0]  # size of data set
    n = int(N/k)        # size of validation set
    offset = 0          # offset index
    performance = 0     # final performance
    for i in range(k):
        
        vset   = train[offset:offset+n]                      # 1/5 validation set
        vset_l = label[offset:offset+n]                      # label for validation set 
        tset   = np.r_[ train[0:offset], train[offset+n:] ]  # 4/5 training set
        tset_l = np.r_[ label[0:offset], label[offset+n:] ]  # label for training set
        
        clf = svc_train(tset, tset_l, C, gamma)              # train the SVC model
        prediction = clf.predict(vset)                       # prediction
        offset += n

        miss = np.count_nonzero(prediction - vset_l)         # count wrong prediction 
        accuracy = 1 - miss/n                                # accuracy
        performance += accuracy
        #print("Accuracy of the {}th validation set: {}%"
        #      .format(i,round(100*accuracy,4)))              
        
    performance /= k                                         # Average total performance 
    #print("The final performance is: {}%".format(round(100*performance,4)))    
        
    return performance

# bayesian optimization
def bayesian_opt(train,label, n_init=1, maxiter=10, opt="GD", verbose=0, acq_funct=1):
    '''
    Parameters
    ----------
    train,label: input data set
    n_init : integer, optional
        number of starting point. The default is 5.
    maxiter: integer, optional
        maximum number of optimization iterations. The default is 10
    opt: string, optional
        options of optimization algorithms
    verbose: integer, optional 
        whether output visualization data. The default is 0 standing for no visualization
        specify verbose (1,...maxiter) to get the visualization data under the verbose-th iteration 
        
    Returns
    -------
    acq_map, min_path, cvg_hist, hlist, best_h, y_best
    acq_map: shape of the acq function at the i-th iteration (specified by verbose)
    min_path: the corresponding minimization path
    hlist: all points found by BO
    best_h: the best point which can achive the highest accuracy
    y_best: the highest accuracy found by BO
    '''
    
    ''' cross validation from sklearn library '''
    def G(C,gamma):
        '''
        Parameters  args = [C,gamma]
        ----------
        C : floating point number      
            Regularization parameter.
        Gamma : floating point number
                Kernel scaling parameter.  
        
        Returns
        -------
        performance: floating point number
            The final performance of the model
        '''
        k=5
        C = 10**C
        gamma = 10**gamma        
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel='rbf',C=C,gamma=gamma))
        scores = cross_val_score(clf, train, label, cv=k)
        performance = np.average(scores)
        
        return performance
    ''' The first choice of acquisition function '''
    def first_acq(h):
        mean,std = gaussian_model(h)  # gaussian model
        p = norm.cdf(y_best,mean,std) # compute the probability
        return -p

    def second_acq(h):                    
        mean,std = gaussian_model(h)  # gaussian model
        return norm.expect(loc=mean-y_best,scale=std,ub=0)
    
    ''' gaussian kernel. by default gamma=0.01 '''
    def gaussian_kernel(x1,x2,gamma=1e-0):  
        return np.exp(-gamma * np.dot(x1-x2,x1-x2))

    ''' gaussian distribution model '''   
    def gaussian_model(h):   
        k = np.array([gaussian_kernel(i,h) for i in hlist])   # k vector
        ksig = np.dot(k, np.linalg.inv(sigma))              
        mean = np.dot(ksig, y)                                # expectation of normal ditribution
        std  = np.sqrt(gaussian_kernel(h,h)-np.dot(ksig,k))   # standard deviation
        return mean,std
    
    ''' output c,gamma meshgrid and corresponding acq function value '''
    def acq_visual(n1=50,n2=50):
        c_grid = np.linspace(0, 9, n1)                        # C grid
        gamma_grid = np.linspace(-10, 0, n2)                  # gamma grid 
        C_grid, Gamma_grid = np.meshgrid(c_grid,gamma_grid)   # meshgrid of [C,gamma]
        acq_mesh = {"C":C_grid,"gamma":Gamma_grid,"acq_func":np.zeros([n2,n1])}    # dictionary {C,gamma,acq_func}           
        # construct the meshgrid of acquisition function
        for i in range(n2):
            for j in range(n1):
                acq_mesh["acq_func"][i,j] = acq( np.array([c_grid[j],gamma_grid[i]]) )                      
        return acq_mesh
    
    ''' initial settings '''    
    if opt == "MBFGS":
        minimum = MBFGS             # MBFGS method
        if acq_funct == 1:          #seting acquisition function
            acq = first_acq
            np.random.seed(5) 
        elif acq_funct == 2:
            acq = second_acq
            np.random.seed(8)       
        gamma = 1e-0                # kernel size
        a_interval = (3,-7)         # selection interval for initial position
        b_interval = (5,-3)         # selection interval for initial position
    else:
        print("No matched optimization method!")
        return None 
    hlist = np.random.uniform(low=(0,-10),high=(9,0),size=(n_init,2))  # uniform dist (C,gamma) = [1,1e9]x[1e-10,1]  
    t = n_init                      # set t = n_init 
    acq_map = None                  # initializae acq_map
    cvg_hist = []                   # convergence history
    
    y = np.array([ G(i[0],i[1]) for i in hlist ])  # initial y list
    sigma = np.zeros([t,t])                        # sigma matrix
    for i in range(t):
        for j in range(t):
            sigma[i,j] = gaussian_kernel(hlist[i],hlist[j],gamma) # initial sigma matrix 
    
    ''' iteration '''
    while(t<maxiter):
        y_best = max(y)             # best y (performance) so far        
        cvg_hist.append(y_best)         
        print("y_best",y_best)
        
        initial_pos = np.random.uniform(low=a_interval,high=b_interval)  # initial position
        if t == verbose:
            acq_map = acq_visual(70,70)                          # visualization of acquisition function
            candidate, min_path = minimum(acq, initial_pos, verbose=1)
        else:
            candidate = minimum(acq, initial_pos)
        
        print("{}th iteration, candidate: {}".format(t,candidate))
        y = np.append(y, G(candidate[0],candidate[1]))      # y list -> new y list
        hlist = np.vstack([hlist,candidate])                # hlist -> new hlist
        K = np.array([gaussian_kernel(i, candidate) for i in hlist]) # K list
        sigma = np.r_[sigma,[K[:-1]]]                       # sigma -> add row
        sigma = np.c_[sigma, K]                             # sigma -> add column 
        t += 1
        
    best_h = hlist[np.argmax(y)]    
    return acq_map, min_path, cvg_hist, hlist, best_h, y_best

def run_mbfgs_bo(maxiter,verbose,acq_choice):
    data = load_data(dir_path+'\heart.csv')                 # load .csv
    train,label = data_preprocessing(data)                  # derive training data and label
    acq_map, min_path, cvg_hist, hlist, best_h, y_best = bayesian_opt(train,label,maxiter=maxiter,opt="MBFGS",verbose=verbose,acq_funct=acq_choice)    # bayesian optimization
    return acq_map, min_path, cvg_hist, hlist, best_h, y_best