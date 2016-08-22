#######################################################################################################
#
# Module : YangCuckooSearch.py
#
# Contains : Functions and methods for the Cuckoo Search Algorithm developed by Yang and Deb to perform
#            non-linear constrainted optimization problems.  The algorithm is given in "Nature-Inspired
#            Optimization Algorithms" Modifcations were made to the data structure, which shouldn't impact 
#            results.
#
# Author : James Bevins
#
# Last Modified: 28Oct15
#
#######################################################################################################

import ObjectiveFunctions as of
import SamplingMethods as sm
import CSLibrary as csl
import numpy as np
import copy as cp
import random as r
import scipy
from scipy import special
import time

#---------------------------------------------------------------------------------------#
def CS(func,lb,ub,S):
    """
    Perform a cuckoo search optimization (CS) based on algorithm in "Nature-Inspired Optimization Algorithms" but modified:
    1) Convergence criteria
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========
   
    Returns
    =======
    timeline : list
        Storage list for best design event objects vs generation for new optimal designs 
    """
    
    start_time=time.time()     #Start Clock
    timeline=[]                #List of history objects
    nests=[]                   #List of nest objects
    
    # Check and establish variable bounds
    assert len(lb)==len(ub), 'Lower and upper-bounds must be the same length'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
        
    # Check for objective function(s) 
    assert hasattr(func, '__call__'), 'Invalid function handle'
    obj = lambda x: func(x)   
    
    # Set seed to aid in debug mode
    if S.d:
        np.random.seed(42)
        r.seed(100)
        
    #Initialize nests with random initial solutions
    init=sm.Initial_Samples(lb,ub,S.s,S.c,S.d)
    for p in range(0,S.c,1):
        nests.append(csl.Nest(5E20,init[p]))  
    if S.d:
        print 'Initial nests:' 
        for i in range(S.c) :    
            print "Nest ",i,": Fitness=",nests[i].f,", Design=",nests[i].d
    
    #Calculate initial fitness values
    (nests,timeline)=csl.Get_Best_Nest(obj,nests,nests,timeline,S)
    
    # Iterate until termination criterion met
    converge = False
    while timeline[-1].g <= S.gm and timeline[-1].e <= S.em and converge==False:
        
        #Global search using Levy flights and evaluate fitness
        new_nests=Get_Nests(nests,lb,ub,S)
        (nests,timeline)=csl.Get_Best_Nest(obj,nests,new_nests,timeline,S,random_replace=False)    
                
        #Local search using discovery and randomization
        new_nests=Empty_Nests(nests,lb,ub,S)
        (nests,timeline)=csl.Get_Best_Nest(obj,nests,new_nests,timeline,S,random_replace=False)
                
        #Test generational convergence
        if timeline[-1].g > S.sl:
            if timeline[-1].g > timeline[-2].g+S.sl:
                converge=True
                print "Generational Stall", timeline[-1].g,timeline[-2].g 
            elif (timeline[-2].f-timeline[-1].f)/timeline[-2].f < S.ct:
                if timeline[-1].g > timeline[-2].g+S.sl:
                    converge=True
                    print "Generational Convergence"
                
        # Test fitness convergence
        if S.of==0.0:
            if timeline[-1].f<S.ot:
                converge=True
                print "Fitness Convergence"                
        elif abs((timeline[-1].f-S.of)/S.of) <= S.ot:
            converge=True
            print "Fitness Convergence"
                    
    #Determine execution time
    print "Program execution time was %f."  % (time.time() - start_time)
    
    return timeline

#---------------------------------------------------------------------------------------#
def Get_Nests(nests,lb,ub,S):
    """
    Generate new nests using Levy flights
   
    Parameters
    ==========
    nests : list of nest objects
        The current nests representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
   
    Returns
    =======
    nests : array
        The new nests representing system designs
    """
    
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Get_Nests function.'
    assert len(lb)==len(nests[0].d), 'Bounds and best nest have different #s of design variables in Get_Nests function.'
    
    tmp=[]
    
    #Levy coefficient
    denom=scipy.special.gamma((1+S.a)/2.)*S.a*2**((S.a-1)/2.)
    sigma=((scipy.special.gamma(1+S.a)*np.sin(np.pi*S.a/2.))/denom)**(1./S.a)
    
    #Simple Levy flights
    for p in range(0,len(nests),1):
            
        # Make a copy of nest for local manipulation
        tmp=nests[p].d
        
        #Levy flight via Mantegna's algorithm
        u=np.random.randn(tmp.size)*sigma 
        v=np.random.randn(tmp.size)
        step=u/np.absolute(v)**(1/S.a)
        
        #(s-best) means that the best solution remains unchanged
        stepsize=S.ss/S.sf*step*(tmp-nests[0].d)
        tmp=tmp+stepsize*np.random.randn(tmp.size)
        if S.d:
            print "The proposed #%d nest, pre-boundary application, is: " %p
            print tmp 
        #Built nest applying variable boundaries
        nests[p].d=csl.Simple_Bounds(tmp,lb,ub,debug=S.d) 
        if S.d:
            print "The proposed #%d nest, post-boundary application, is: " %p
            print nests[p].d 
        
    if S.d:
         for i in range(len(nests)) : 
            print "The new proposed nests are: ", nests[i].d 
        
    return nests
   
#---------------------------------------------------------------------------------------#           
def Empty_Nests(nests,lb,ub,S):
    """
    Generate new nests by emptying pa nests
   
    Parameters
    ==========
    nest : array
        The current system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
   
    Returns
    =======
    new_nest : array
        The new system designs after pa nest were abandoned
    """
    
    assert len(nests[0].d)==len(lb), 'Nest and best have different #s of design variables in Empty_Nests function.'
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Empty_Nests function.'
    assert S.pa>0 and S.pa <=1, 'The probability that a nest is discovered must exist on (0,1]'
    
    new_nests=[]
    
    #Discover pa of worst nest; K is a status vector to see if discovered
    K=np.random.rand(len(nests),len(nests[0].d))>S.pa
        
    #Bias the discovery to the worst fitness solutions
    nestn1=cp.copy(np.random.permutation(nests))
    nestn2=cp.copy(np.random.permutation(nests))     
    
    #New solution by biased/selective random walks
    r=np.random.rand()
    for j in range(0,len(nests),1):
        n=np.array((nestn1[j].d-nestn2[j].d))
        step_size=r*n
        s=nests[j].d+step_size*K[j,:]
        new_nests.append(csl.Nest(nests[j].f,csl.Simple_Bounds(s,lb,ub)))
        
    if S.d:
        print "\nAt end of function Empty_Nests:"
        print "The status vector is :", K
        print "The step size is :", step_size
        for i in range(len(nestn1)):
            print "The first permutation is ", nestn1[i].f,nestn1[i].d
        for i in range(len(nestn2)):
            print "The second permutation is ", nestn2[i].f,nestn2[i].d
        for i in range(len(new_nests)):
            print "The new nests #",i,"is ", new_nests[i].d
    return new_nests  

#---------------------------------------------------------------------------------------#
