#######################################################################################################
#
# Module : BevinsCuckooSearch.py
#
# Contains : Functions and methods for a basic modification of the Cuckoo Search Algorithm developed by 
#            Yang and Deb to perform non-linear constrainted optimization problems.  The original  
#            algorithm is given in "Nature-Inspired Optimization Algorithms". Modifications include:
#            1) Convergene criteria
#            2) Levy algorithm
#            3) 1 random global search with random global replacement
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
import random as r
import math as m
import copy as cp
import time

#---------------------------------------------------------------------------------------#
def CS(func,lb,ub,S):
    """
    Perform a cuckoo search optimization (CS) based on algorithm in 
    "Nature-Inspired Optimization Algorithms" but modified:
    1) Convergene criteria
    2) Levy algorithm
    3) 1 random global search with random global replacement
    
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
        
        # Global search using Levy flights and evaluate fitness
        new_nests=Get_Nests(nests,lb,ub,S)
        (nests,timeline)=csl.Get_Best_Nest(obj,nests,new_nests,timeline,S,random_replace=True)
        
        
        # Perform regional search by mutating top subset of cuckoos
        (nests,timeline) = Mutate_Nests(obj,nests,timeline,S)
                                     
        # Local search using discovery and randomization
        new_nests=Empty_Nests(nests,lb,ub,S)
        (nests,timeline)=csl.Get_Best_Nest(obj,nests,new_nests,timeline,S,random_replace=False)  
        
        # Test generational convergence
        if timeline[-1].g > S.sl:
            if timeline[-1].g > timeline[-2].g+S.sl:
                converge=True
                print "Generational Stall", timeline[-1].g,timeline[-2].g 
            elif timeline[-2].f==0:
                if S.of==0:
                    converge=True
                    print "Generational Convergence"
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
    tmp : list of nest objects
        The proposed nests representing new system designs
    """
    
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Get_Nests function.'
    assert len(lb)==len(nests[0].d), 'Bounds and nests have different #s of design variables in Get_Nests function.'   
    assert S.pt>=0 and S.pt <=1, 'The probability that a nest is used for global search must exist on (0,1] & =%d' %S.pt
    
    tmp=[] # Local copy of nest that is modified
            
    # Determine step size using Levy Flight
    step=sm.Levy(len(nests[0].d),len(nests),alpha=S.a,gamma=S.g,n=S.n,debug=S.d) 
    if S.d:
        print "\nIn function Get_Nests,"
        print "The steps are:", step
        print "The number of Levy flights are: ", int(S.c*S.pt)
        print S.a,S.g,S.n,S.ss/S.sf
    
    # Perform global search from pc nests
    used=[]
    for i in range(int(S.c*0.4)):
        # Select random nest and make a local copy
        p=int(np.random.rand()*S.c)
        while p in used:
            p=int(np.random.rand()*S.c)
        used.append(p)
        tmp.append(cp.deepcopy(nests[p]))
        
        #Calculate Levy flight
        stepsize=1.0/S.sf*step[p,:]#*(tmp[i].d-nests[0].d)   
        tmp[i].d=tmp[i].d+stepsize#*np.random.randn(len(tmp[0].d))  
        if S.d:
            print "The old randomly chosen nest is", nests[p].d
            print "The step sizes are: ", stepsize
            print "The proposed #",i,"nest, pre-boundary application, is: ",tmp[i].d
        
        #Built nest applying variable boundaries 
        tmp[i].d=csl.Rejection_Bounds(nests[p].d,tmp[i].d,stepsize,lb,ub,S,debug=S.d) 
        if S.d:
            print "The proposed #",i,"nest, pre-boundary application, is: ",tmp[i].d

    if S.d:
        print "The new proposed nests are: ",
        for i in range(len(tmp)) : 
            print tmp[i].d
        
    return tmp
            
#---------------------------------------------------------------------------------------#               
def Empty_Nests(nests,lb,ub,S):
    """
    Generate new nests by emptying pa nests
   
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
    tmp : list of nest objects
        The proposed nests representing new system designs
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
        tmp=nests[j].d+step_size*K[j,:]
        new_nests.append(csl.Nest(nests[j].f,csl.Simple_Bounds(tmp,lb,ub,debug=S.d)))
        
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
def Mutate_Nests(func,nests,timeline,S):  
    """
    Generate new nests using distance based mutation strategies on the top S.pt nests.  
    Idea adapted from Walton "Modified Cuckoo Search: A New Gradient Free Optimisation Algorithm"
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    nests : list of nest objects
        The current nests representing system designs
    timeline : list
        Storage list for best design event objects vs generation for new optimal designs 
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
   
    Returns
    =======
    nests : list of nest objects
        The nests representing new system designs
    timeline : list
        Storage list for best design event objects vs generation for new optimal designs 
    """
    
    assert 0<=S.pt<=1, 'pt must be between 0 and 1 inclusive in Mutate_Nests function.'
    
    # Initialize variables
    golden_ratio=(1.+m.sqrt(5))/2.  # Used to bias distance based mutation strategies
    top_nests=[]
    top_nests.append(cp.deepcopy(nests[0]))
    feval=0
    dx=np.zeros_like(nests[0].d)
    
    # Mutate top S.pt nests with the nest below
    for i in range(0,int(S.c*S.pt),1):
        top_nests.append(cp.deepcopy(nests[i+1]))
        dx=abs(top_nests[i].d-top_nests[i+1].d)/golden_ratio
        top_nests[i].d=top_nests[i+1].d+dx
        
        # Evaluate fitness of mutated nest and compare to old to see if improvement
        (fnew,gnew)=func(top_nests[i].d)
        feval+=1
        if fnew < nests[i+1].f:
            nests[i+1]=cp.deepcopy(top_nests[i])
    
    #Update timeline with new fitness values
    timeline[-1].e+=feval
                    
    if S.d:
        print "\nAt end of Mutate_Nests, the new proposed nests are: "
        for i in range(0,len(nests),1): 
            print "New nest #", i,":",nests[i].d 
        
    return nests, timeline