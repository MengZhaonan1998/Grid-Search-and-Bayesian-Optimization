# -*- coding: utf-8 -*-
"""
<Bayesian Optimization>-interface
Created on Sat Dec 31 13:33:06 2022
@author: Zhaonan Meng
Please run this program directly!
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))  # getting current directory
sys.path.append(dir_path)
from GD_BO import run_gd_bo          # import gradient-descent-based bayesian optimization
#from MBFGS_BO import run_mbfgs_bo    # import MBFGS-based bayesian optimization
from MBFGS_BO import run_mbfgs_bo
# plot the shape of acquisition function
def display_acq(acq_map,min_path):
    x = np.array(acq_map['C'])
    y = np.array(acq_map['gamma'])
    z = np.array(acq_map['acq_func'])

    plt.figure(figsize=(9,6.8))
    plt.contourf(x, y, z, 20, cmap='YlGnBu_r')
    plt.colorbar()
    plt.xlabel(r"$log_{10}C$")
    plt.ylabel(r"$log_{10}\gamma$")
    
    min_path = min_path[ (min_path[:,0]>0) & (min_path[:,0]<9) ]
    min_path = min_path[ (min_path[:,1]>-10) & (min_path[:,1]<0) ]
    plt.plot(min_path.T[0],min_path.T[1],c='indianred',linewidth=2.5,label='minimization path')
    plt.text(min_path[-1][0],min_path[-1][1],'local minimum',fontsize=13,horizontalalignment='left',verticalalignment='top',c='red')
    plt.legend()
    plt.show()
    
# plot the best performance every iteration    
def display_cvghist(cvg_hist,n):
    plt.figure(figsize=[8,5.5])
    plt.plot(np.linspace(1,n,n),cvg_hist[:n])
    plt.xlabel('iteration')
    plt.ylabel('performance of cross validation')
    plt.xticks(np.arange(1,n,4))
    plt.grid()
    plt.show()

# scatter plot of points found by BO inside the domain    
def display_points(hlist,best_h):
    plt.figure(figsize=[8,7])
    plt.scatter(hlist.T[0],hlist.T[1],s=6)
    plt.scatter(best_h[0],best_h[1],s=12,c='red',label='best configuration')
    plt.xlabel(r"$log_{10}C$")
    plt.ylabel(r"$log_{10}\gamma$")
    plt.xlim(0,9)
    plt.ylim(-10,0)
    plt.xticks(np.linspace(0,9,10))
    plt.yticks(np.linspace(-10,0,11))
    plt.legend()
    plt.grid()
    plt.show()
        
# all results (including all figures)
def result(option_bo):
    if option_bo == 1:
        run_bo = run_gd_bo
        acq_no = 1
        n = 20     # this n is for display_cvghist(cvg_hist,n) 
    elif option_bo == 2:
        run_bo = run_mbfgs_bo
        acq_no = 1
        n = 50      # this n is for display_cvghist(cvg_hist,n) 
    elif option_bo == 3:
        run_bo = run_gd_bo
        acq_no = 2
        n = 20     # this n is for display_cvghist(cvg_hist,n)
    elif option_bo == 4:
        run_bo = run_mbfgs_bo
        acq_no = 2
        n = 50      # this n is for display_cvghist(cvg_hist,n) 

    start_t = time.time()
    acq_map, min_path, cvg_hist, hlist, best_h, y_best = run_bo(maxiter=100,verbose=25,acq_choice=acq_no)
    end_t = time.time()
    print("-----------------------------------------------")
    print("Bayesian optimization completed, taking {} seconds".format(end_t-start_t))
    print("The best accuracy is {}. And the corresponding log10(C,gamma) is {}".format(y_best,best_h))
    
    display_cvghist(cvg_hist,n)      # display the the best performance every iteration
    display_acq(acq_map,min_path)    # display the shape of acquisition function
    display_points(hlist,best_h)     # display all points found by bayesian opt
    
print("Welcome to the second part of our project 2: Bayesian Optimization!")
print("This part is contributed by Zhaonan Meng and Ditra Matin.")
print("Here we offer two types of Bayesian Optimization:")
print("    1: with the first acquisition function and classical gradient descent optimization method")
print("    2: with the first acquisition function and Quasi-Newton MBFGS optimization method")
print("    3: with the second acquisition function and classical gradient descent optimization method (WARNING: very slow, may take up to 1.5 hours)")
print("    4: with the second acquisition function and Quasi-Newton MBFGS optimization method")
print("They have some difference in performance such as efficiency and best misclassification error found.")
print("Please select one of them by typing 1, 2, 3, xor 4 and then you can see all results directly")
choice = input()
if choice == '1':
    result(1)
elif choice == '2':
    result(2)
elif choice == '3':
    result(3)
elif choice == '4':
    result(4)
else:
    print("wrong input! please run the program again and select 1, 2, 3, xor 4!")


