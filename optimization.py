# -*- coding: utf-8 -*-
"""
Optimization Algorithms 
Created on Thu Dec  1 16:00:26 2022
@author: Zhaonan Meng, Ditra Matin

This .py file provides several popular optimization methods such as
gradient descent method and so on.
"""
import numpy as np 

# classical gradient descent with bisection line search
def gradient_descent(f,x,eps=1e-5,maxiter=50,verbose=0):
    '''
    Parameters
    ----------
    f : function def 
        function you want to minimize 
    x : double floating array 
        starting point for the algorithm
    eps : small double floating number 
        tolerance for stopping criteria
    maxiter: integer
        maximum number of iterations 
    verbose: integer
        whether output x of every iteration or not

    Returns
    -------
    x : double floating array
        (local) minimal point
    '''
    if verbose == 0:
        for i in range (maxiter):
            gf = gradient(f,x)
            
            if np.linalg.norm(gf) < eps:
                return x
            
            g  = g_func(f,x)
            t  = bisection_ls(g, b=1)
            x  = x - t * gf
            #print("f(x_k):",f(x))

        print("maximum iterations reached!")
        return x
    
    else:
        xlist = x
        for i in range (maxiter):
            gf = gradient(f,x)
            
            if np.linalg.norm(gf) < eps:
                return x, xlist
            
            g  = g_func(f,x)
            t  = bisection_ls(g, b=1)
            x  = x - t * gf
            
            xlist = np.vstack((xlist, x))
        
        print("maximum iterations reached!")
        return x, xlist

# beyond the gradient method
# simple momentum-based learning
def momentum_grad(f,x,beta=0.5,eps=1e-5,maxiter=30):
    '''
    Parameters
    ----------
    f : function def 
        function you want to minimize 
    x : double floating array 
        starting point for the algorithm
    beta: floating point number
        updating parameter 
    eps : small double floating number 
        tolerance for stopping criteria
    maxiter: integer
        maximum number of iterations 

    Returns
    -------
    x : double floating array
        (local) minimal point
    '''
    d = np.zeros(len(x))
    for i in range (maxiter):
        #print("f(x)=",f(x))
        gf = gradient(f,x)
        
        if np.linalg.norm(gf) < eps:
            return x
        
        g  = g_func(f,x)
        t  = bisection_ls(g, b=1)
        d  = beta * d - t * gf
        x  = x + d
        #print("f(x_k):",f(x))
        
    print("maximum iterations reached!")
    return x

# adaptive gradient descent
def ada_grad(f,x,alpha=1,eps=1e-5,maxiter=30):
    '''
    Parameters
    ----------
    f : function def 
        function you want to minimize 
    x : double floating array 
        starting point for the algorithm
    alpha: floating point number
        updating parameter
    eps : small double floating number 
        tolerance for stopping criteria
    maxiter: integer
        maximum number of iterations 

    Returns
    -------
    x : double floating array
        (local) minimal point
    '''
    A = np.zeros(len(x))  # initialize A
    for i in range(maxiter):
        gf = gradient(f,x)
        
        if np.linalg.norm(gf) < eps:
            return x
         
        A = A + gf * gf
        x = x - alpha * gf / np.sqrt(A)

    print("maximum iterations reached!")
    return x

def linesearch_LF(f,x,rho,p,sigma,g,maxj):
    '''
    https://www.sciencedirect.com/science/article/pii/S0377042700005409
    Parameters
    ----------
    f : function def 
        function you want to minimize 
    p : double floating array 
        search direction
    rho   : floating point number
            step size parameter
    sigma : small double floating number 
            parameter for step size search
    maxj  : maximum number of iterations 

    Returns
    -------
    j : integer
        power of rho in MBFGS algorithm
    '''
    for j in range(maxj):
        if f(x + (rho**j)*p) <= f(x) + sigma*(rho**j)*np.dot(g,p):
            return j
    return j

def MBFGS(f,x,eps=1e-5,maxiter=50,verbose=0):
    ''''
    https://www.sciencedirect.com/science/article/pii/S0377042700005409
    Parameters
    ----------
    f : function def 
        function you want to minimize 
    x : double floating array 
        starting point for the algorithm
    eps : small double floating number 
        tolerance for stopping criteria
    maxiter: integer
        maximum number of iterations 

    Returns
    -------
    x : double floating array
        (local) minimal point
    '''
    if verbose == 0:
        B = np.eye(len(x))
        rho = 0.01
        sigma = 0.01
        for i in range(maxiter):
            g = gradient(f,x)
            
            if np.linalg.norm(g) < eps:
                return x
            
            p = np.linalg.solve(B,-g)
            lam = rho**(linesearch_LF(f,x,rho,p,sigma,g,maxj=100))
            s = lam*p
            x_new = x + s
            gamma = gradient(f,x_new) - g
            t = 1 + max((-np.dot(gamma,s)/(np.linalg.norm(s)**2)),0)
            y = gamma + t*np.linalg.norm(g)*s
            B = B - (B@np.outer(s,s.T)@B)/(s.T@B@s) + (np.outer(y,y.T))/(y.T@s)
            
            if x_new[0] < 0 or x_new[0] > 9:
                x_new[0] = x[0]
            if x_new[1] < -10 or x_new[1] > 0:
                x_new[1] = x[1]
            
            x = x_new
        return x
    else:
        xlist = x
        B = np.eye(len(x))
        rho = 0.01
        sigma = 0.01
        for i in range(maxiter):
            #print("f(x_k):",f(x))
            g = gradient(f,x)
            
            if np.linalg.norm(g) < eps:
                return x,xlist
            
            p = np.linalg.solve(B,-g)
            lam = rho**(linesearch_LF(f,x,rho,p,sigma,g,maxj=100))
            s = lam*p
            x_new = x + s
            gamma = gradient(f,x_new) - g
            t = 1 + max((-np.dot(gamma,s)/(np.linalg.norm(s)**2)),0)
            y = gamma + t*np.linalg.norm(g)*s
            B = B - (B@np.outer(s,s.T)@B)/(s.T@B@s) + (np.outer(y,y.T))/(y.T@s)
            
            x = x_new
            xlist = np.vstack((xlist, x))
        return x,xlist

# root mean square propagation
# doesn't work so far
def rmsprop(f,x,alpha=1,rho=0.5,eps=1e-5,maxiter=30):
    '''
    Parameters
    ----------
    f : function def 
        function you want to minimize 
    x : double floating array 
        starting point for the algorithm
    alpha, rho: floating point number
        updating parameter
    eps : small double floating number 
        tolerance for stopping criteria
    maxiter: integer
        maximum number of iterations 

    Returns
    -------
    x : double floating array
        (local) minimal point
    '''
    A = np.zeros(len(x))  # initialize A
    for i in range(maxiter):
        gf = gradient(f,x)
        
        if np.linalg.norm(gf) < eps:
            return x
        
        A = rho * A + (1-rho) * gf * gf
        x = x - alpha * gf / np.sqrt(A)

    print("maximum iterations reached!")
    return x

# adaptive moment estimation
# doesn't work so far
def adamoment(f,x,alpha=1,rho=0.5,eps=1e-5,maxiter=30):
    '''
    Parameters
    ----------
    f : function def 
        function you want to minimize 
    x : double floating array 
        starting point for the algorithm
    alpha, rho: floating point number
        updating parameter
    eps : small double floating number 
        tolerance for stopping criteria
    maxiter: integer
        maximum number of iterations 

    Returns
    -------
    x : double floating array
        (local) minimal point
    '''
    F = np.zeros(len(x))
    A = np.zeros(len(x))  
    for i in range(maxiter):
        gf = gradient(f,x)
        
        if np.linalg.norm(gf) < eps:
            return x
        
        A = rho * A + (1-rho) * gf * gf
        F = rho * F + (1-rho) * gf 
 
        x = x - alpha * F / np.sqrt(A)

    print("maximum iterations reached!")
    return x

# compute the gradient of objective function f in certain point x
def gradient(f,x,eps=1e-6):
    '''
    Parameters
    ----------
    f : function def 
        Objective function 
    x : double floating array 
    eps : small double floating number 
        step size for differentiation

    Returns
    -------
    g : double floating array
        gradient of f(x)
    '''    
    if type(x)==float:
        # univariate
        g = (f(x + eps) - f(x)) / eps           # forward difference
    else:  
        # multivariate  
        x = x.astype('float64')                                  
        n = len(x)                              # size of x  
        g = np.zeros(n)                         # gradient 
        for i in range(n):
            perturb = np.array([j for j in x])  # donnot use perturb=x! note the array pointer in python
            perturb[i] += eps                   # forward difference
            g[i] = ( f(perturb) - f(x) ) / eps  # gradient 
    return g

# compute the second-order gradient of objective function f in certain point x
def ggradient(f,x,eps=1e-3):
    '''
    Parameters
    ----------
    f : function def 
        Objective function 
    x : double floating array 
    eps : small double floating number 
        step size for differentiation

    Returns
    -------
    gg : double floating array
        second-order gradient of f(x)
    '''    
    if type(x)==float:
        # univariate
        gg = (f(x + eps) - 2*f(x) + f(x - eps)) / (eps * eps)    # D+D- difference
    else:  
        # multivariate  
        x = x.astype('float64')                                  
        n = len(x)                              # size of x  
        gg = np.zeros(n)                         # gradient 
        for i in range(n):
            fperturb = np.array([j for j in x])  # donnot use perturb=x! note the array pointer in python
            bperturb = np.array([j for j in x])  
            fperturb[i] += eps                   # D+D-
            bperturb[i] -= eps                   
            gg[i] = ( f(fperturb) - 2*f(x) + f(bperturb) ) / (eps*eps)  # second order gradient 
    return gg

# bisection algorithm for line search 
def bisection_ls(g,b,eps=1e-6):
    '''
    Parameters
    ----------
    g : function def 
        function of t(stepsize) g(t)=f(x-t*f'(x)) 
    b : double floating array
        right starting point (>0)
    eps : small double floating number 
        tolerance for bisection

    Returns
    -------
    t : double floating number
        proper stepsize
    '''
    a = 0
    while abs(a-b)>eps:
        c = (a+b)/2
        gd = gradient(g, c)  # derivative of g(t)
        if gd > 0:
            b = c
        else:
            a = c
    return a

# backtracking step size selection
# there are some problems ...
def backtracking(f,x,s,alpha=0.5,beta=0.5):
    '''
    Parameters
    ----------
    f : function def 
        objective function 
    x : double floating array
        x_k
    s : double floating array
        starting step size
    alpha, beta: floating point number [0,1]

    Returns
    -------
    t : double floating number
        proper stepsize
    '''
    t = s
    g = gradient(f,x)
    d = g/np.linalg.norm(g)
    while True:
        diff = f(x) - f(x+t)
        
        if diff < -alpha * t * np.dot(g,d):
            t *= beta
        else:
            return t
        
# compute g(t)=f(x-t*f'(x)) 
def g_func(f,x):
    '''
    Parameters
    ----------
    f : function def
        objective function
    x : double floating array
        x_k

    Returns
    -------
    g : function def
        g(t)
    '''
    fg = gradient(f, x)       # gradient of f(x_k)   
    def g(t):
        return f(x - t * fg)  # g(t)=f(x_k-t*f'(x_k)) 
    return g


def f(x):
    return 1.8*x[0]*x[0] + 2.42*x[1]*x[1] + 2*x[1] # 3.1*x[1]*x[1] - 2.6
