#######################################################################################################
#
# Module : SamplingMethods.py
#
# Contains : Different methods to perform phase space sampling and random walks and visualization tools
#
# Author : James Bevins
#
# Last Modified: 23Nov16
#
#######################################################################################################

import scipy
from scipy import special
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import ObjectiveFunctions as of
import numpy as np
import copy as cp
import math
import argparse
import random
from pyDOE import *

#---------------------------------------------------------------------------------------#
def Initial_Samples(lb,ub,method,n=25,debug=False):  
    """
    Generate a set of samples in a given phase space
   
    Parameters
    ==========
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    method : string    
        String representing the chosen sampling method
   
    Optional
    ========   
    n : int
        The number of samples to be generated.  Ignored for nolh algorithms. (Default=25)
   
    Returns
    =======
    s : array
        The list of coordinates for the sampled phase space
    """  
    
    assert len(ub)==len(lb), 'Boundaries best have different # of design variables in Initial_Samples function.'
    assert len(ub)>=2 and len(ub)<=29, 'The Phase space dimensions are outside of the bounds for Initial_Samples.'
    assert method=='random' or method=='nolh' or method=='nolh-rp' or method=='nolh-cdr' or method=='lhc', \
           'An invalid method was specified for the initial sampling.'
    
    for case in of.switch(method):
        if case('random'): 
            s=np.zeros((n,len(lb)))
            for i in range(0,n,1):
                s[i,:]=(lb+(ub-lb)*np.random.rand(len(lb)))  
            break
        # Standard nearly-orthoganal latin hypercube call   
        if case('nolh'):
            dim = len(ub)
            m, q, r = params(dim)
            conf = range(q)
            if r!=0:
                remove = range(dim - r, dim)
                nolh = NOLH(conf, remove)
            else:
                nolh = NOLH(conf)
            s=np.array([list(lb+(ub-lb)*nolh[i,:]) for i in range(len(nolh[:,0]))])
            break           
        # Nearly-orthoganal latin hypercube call with random permutation for removed colummns  
        if case('nolh-rp'):
            dim = len(ub)
            m, q, r = params(dim)
            print r
            conf = random.sample(range(q), q)
            if r!=0:
                remove = random.sample(range(q-1), r)
                print remove
                nolh = NOLH(conf, remove)
            else:
                nolh = NOLH(conf)
            s=np.array([list(lb+(ub-lb)*nolh[i,:]) for i in range(len(nolh[:,0]))])
            break          
        # Nearly-orthoganal latin hypercube call with Cioppa and De Rainville permutations 
        if case('nolh-cdr'):
            dim = len(ub)
            m, q, r = params(dim)
            (conf,remove)=Get_CDR_Permutations(len(ub))
            if remove!=[]:
                nolh = NOLH(conf, remove)
            else:
                nolh = NOLH(conf)
            s=np.array([list(lb+(ub-lb)*nolh[i,:]) for i in range(len(nolh[:,0]))])
            break
        # Latin hypercube sampling 
        if case('lhc'):
            tmp=lhs(len(lb),samples=n,criterion="center") #Other valid criterion are 'corr','center','maximum','centermaximum'
            s=np.array([list(lb+(ub-lb)*tmp[i,:]) for i in range(len(tmp[:,0]))])
            break
        if case(): 
            print "Somehow you evaded my assert statement - good job!  However, you still need to use a valid method string."
    
    return s

#---------------------------------------------------------------------------------------#
def Plot_Samples(s):  
    """
    Plot the first 2 and 3 dimensions on the sample distribution.  
   
    Parameters
    ==========
    s : array
        The list of coordinates for the sampled phase space
   
    Optional
    ========
   
    Returns
    =======
    None
    """  
    
    assert len(s[0,:])>=2, 'The Phase space dimensions are less than two.  Need at least two to plot.'
    
    fig = plt.figure(1)
    if len(s[0,:])>=3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(s[:,0],s[:,1],s[:,2])
    fig = plt.figure(2)
    plt.scatter(s[:,0],s[:,1])
    plt.show()
    return

#---------------------------------------------------------------------------------------#
def Levy(nc,nr=0,alpha=1.5,gamma=1,n=1):
    """
    Generate Levy flights steps
   
    Parameters
    ==========
    nc : int
        The number of columns of Levy values for the return array
  
    Optional
    ========   
    nr : int
        The number of rows of Levy values for the return array
    alpha : scalar
        Levy exponent - defines the index of the distribution and controls scale properties of the stochastic process
        (Default: 1.5)
    gamma : scalar
        Gamma - Scale unit of process for Levy flights (Default: 1)
    n : integer
        Number of independent variables - can be used to reduce Levy flight variance (Default: 1)
   
    Returns
    =======
    z : array
        Array representing the levy flights for each nest
    """
    
    assert alpha > 0.3 and alpha <1.99, 'Valid range for alpha is [0.3:1.99].'
    assert gamma >= 0, 'Gamma must be positive'
    assert n >=1, 'n Must be positive'
        
    # Calculate Levy distribution via Mantegna Algorithm
    # Fast, accurate algorithm for numerical simulation of Levy stable stochastic processes
    invalpha=1./alpha
    sigx=((scipy.special.gamma(1.+alpha)*np.sin(np.pi*alpha/2.))/(scipy.special.gamma((1.+alpha)/2)
                                                                  *alpha*2.**((alpha-1.)/2.)))**invalpha
    if nr!=0:
        v = sigx*np.random.randn(n,nr,nc)/(abs(np.random.randn(n,nr,nc))**invalpha)
    else:
        v = sigx*np.random.randn(n,nc)/(abs(np.random.randn(n,nc))**invalpha)
    kappa = (alpha*scipy.special.gamma((alpha+1.)/(2.*alpha)))/scipy.special.gamma(invalpha) \
         *((alpha*scipy.special.gamma((alpha+1.)/2.))/(scipy.special.gamma(1.+alpha)*np.sin(np.pi*alpha/2.)))**invalpha
    p =  [-17.7767,113.3855,-281.5879,337.5439,-193.5494,44.8754] 
    c = np.polyval(p, alpha)
    w = ((kappa-1.)*np.exp(-abs(v)/c)+1.)*v
    
    if n>1: 
        z = (1/n**invalpha)*sum(w) 
    else:
        z = w 

    z = gamma**invalpha*z
    if nr!=0:
        z=z.reshape(nr,nc)
    else:
        z=z.reshape(nc)

    return z  

#---------------------------------------------------------------------------------------#
def TLF(numRow=1,numCol=1,alpha=1.5,gamma=1.,cut_point=10.):
    
    """
    Samples from a truncated Levy flight distribution (TLF) according to Manegna, "Stochastic Process 
    with Ultraslow Convergence to a Gaussian: The Truncated Levy Flight" to map a levy distribution onto
    the interval [0,1].
   
    Parameters
    ==========
  
    Optional
    ========   
    numRow : integer
        Number of rows of Levy flights to sample (Default: 1)
    numCol : integer
        Number of columns of Levy flights to sample (Default: 1)
    alpha : scalar
        Levy exponent - defines the index of the distribution and controls scale properties of the stochastic process
        (Default: 1.5)
    gamma : scalar
        Gamma - Scale unit of process for Levy flights (Default: 1.)
    cut_point : scalar
        Point at which to cut sampled Levy values and resample
   
    Returns
    =======
    levy : array
        Array representing the levy flights on the interval (0,1)
    """
    
    # Draw numRow x numCol samples from the Levy distribution
    levy=abs(Levy(numRow,numCol)/cut_point).reshape(numRow,numCol)
    
    # Resample values above the range (0,1)
    for i in range(len(levy)):
        for j in range(len(levy[i])):
            while levy[i,j]>1:
                levy[i,j]=abs(Levy(1,1)/cut_point).reshape(1)
            
    return levy

#---------------------------------------------------------------------------------------# 
def NOLH(conf, remove=None):
    """
    This library allows to generate Nearly Orthogonal Latin Hypercubes (NOLH) according to
    Cioppa (2007) and De Rainville et al. (2012) and reference therein.
    https://pypi.python.org/pypi/pynolh
    Constructs a Nearly Orthogonal Latin Hypercube (NOLH) of order *m* from a configuration vector
    *conf*. The configuration vector may contain either the numbers in $[0 q-1]$ or $[1 q]$ where 
    $q = 2^{m-1}$. The columns to be *removed* are also in $[0 d-1]$ or $[1 d]$ where $d = m + 
    \binom{m-1}{2}$ is the NOLH dimensionality.
    
    The whole library is incorporated here with minimal modification for commonality and 
    consolidation of methods. 
    
    Parameters
    ==========
    conf : array
        Configuration vector
   
    Optional
    ========   
    remove : array
        Array containing the indexes of the colummns to be removed from conf vetor
        (Default: NONE)
   
    Returns
    =======
    nolh : array
        Array containing nearly orthogonal latin hypercube sampling.
    """
    
    I = np.identity(2, dtype=int)
    R = np.array(((0, 1),
                     (1, 0)), dtype=int)

    if 0 in conf:
        conf = np.array(conf) + 1

        if remove is not None:
            remove = np.array(remove) + 1


    q = len(conf)
    m = math.log(q, 2) + 1
    s = m + (math.factorial(m - 1) / (2 * math.factorial(m - 3)))
    # Factorial checks if m is an integer
    m = int(m)

    A = np.zeros((q, q, m - 1), dtype=int)
    for i in range(1, m):
        Ai = 1
        for j in range(1, m):
            if j < m - i:
                Ai = np.kron(Ai, I)
            else:
                Ai = np.kron(Ai, R)

        A[:, :, i-1] = Ai
    
    M = np.zeros((q, s), dtype=int)
    M[:, 0] = conf
    
    col = 1
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            if i == 0:
                M[:, col] = np.dot(A[:, :, j-1], conf)
            else:
                M[:, col] = np.dot(A[:, :, i-1], np.dot(A[:, :, j-1], conf))
            col += 1

    S = np.ones((q, s), dtype=int)
    v = 1
    for i in range(1, m):
        for j in range(0, q):
            if j % 2**(i-1) == 0:
                v *= -1
            S[j, i] = v

    col = m
    for i in range(1, m - 1):
        for j in range(i + 1, m):
            S[:, col] = S[:, i] * S[:, j]
            col += 1

    T = M * S
    
    keep = np.ones(s, dtype=bool)
    if remove is not None:
        keep[np.array(remove) - 1] = [False] * len(remove)
    
    return (np.concatenate((T, np.zeros((1, s)), -T), axis=0)[:, keep] + q) / (2.0 * q)

def params(dim):
    """Returns the NOLH order $m$, the required configuration length $q$
    and the number of columns to remove to obtain the desired dimensionality.
    """
    m = 3
    s = 1     #Original version has three here, but this failed each time the # of samples required switched (ie at dim=3,7,11,etc)
    q = 2**(m-1)
    
    while s < dim:
        m += 1
        s = m + math.factorial(m - 1) / (2 * math.factorial(m - 3))
        q = 2**(m-1)

    return m, q, s - dim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Compute a Nearly "
        "Orthogonal Latin hypercube from a configuration vector."))
    parser.add_argument("conf", metavar="C", type=int, nargs="+",
        help="The configuration vector given as a list N1 N2 ... Nm")
    parser.add_argument("-r", "--remove", metavar="R", type=int, nargs="+",
        help="Columns to remove as a list N1 N2 ... Nm")

    args = parser.parse_args()
    print(nolh(conf=args.conf, remove=args.remove))

#---------------------------------------------------------------------------------------#  
def Get_CDR_Permutations(dim):  
    """
    Generate a set of samples in a given phase space
   
    Parameters
    ==========
    dim : integer
        The dimension of the phase space
   
    Returns
    =======
    conf : array
        Configuration vector
    remove : array
        Array containing the indexes of the colummns to be removed from conf vetor
    """  
    
    assert dim>=2 and dim<=29, 'The Phase space dimensions are outside of the bounds for CDR Permutations.' 

    # Permutation and columns to remove given by Cioppa
    C_CONF = {
        2 : ([1, 2, 8, 4, 5, 6, 7, 3], [1, 3, 4, 6, 7]),
        3 : ([1, 2, 8, 4, 5, 6, 7, 3], [1, 2, 3, 6]),
        4 : ([1, 2, 8, 4, 5, 6, 7, 3], [1, 3, 6]),
        5 : ([1, 2, 8, 4, 5, 6, 7, 3], [1, 6]),
        6 : ([1, 2, 8, 4, 5, 6, 7, 3], [1]),
        7 : ([1, 2, 8, 4, 5, 6, 7, 3], [])
    }  
    
    # Permutation and columns to remove given by De Rainville et al.
    EA_CONF = {
        8  : ([4, 14, 1, 2, 16, 13, 5, 8, 12, 9, 6, 7, 11, 3, 15, 10], [1, 3, 10]),
        9  : ([4, 14, 1, 2, 16, 13, 5, 8, 12, 9, 6, 7, 11, 3, 15, 10], [6, 10]),
        10 : ([4, 14, 1, 2, 16, 13, 5, 8, 12, 9, 6, 7, 11, 3, 15, 10], [10]),
        11 : ([4, 14, 1, 2, 16, 13, 5, 8, 12, 9, 6, 7, 11, 3, 15, 10], []),

        12 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8, 24,
               29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4], [2, 4, 5, 11]),
        13 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8, 24,
               29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4], [3, 6, 14]),
        14 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8, 24,
               29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4], [4, 5]),
        15 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8, 24,
               29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4], [6]),
        16 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8, 24,
               29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4], []),

        17 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60], [8, 11, 12, 14, 17]),
        18 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60], [8, 11, 12, 17]),
        19 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60], [10, 15, 22]),
        20 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60], [8, 12]),
        21 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60], [15]),
        22 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60], []),

        23 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [18, 20, 21, 24, 27, 29]),
        24 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [4, 15, 18, 24, 27]),
        25 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [21, 26, 27, 29]),
        26 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [26, 27, 29]),
        27 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [27, 29]),
        28 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [20]),
        29 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [])
    }
    
    # Create dictionary
    CONF = dict()
    CONF.update(C_CONF)
    CONF.update(EA_CONF)
    
    return CONF[dim][0], CONF[dim][1]