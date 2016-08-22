#######################################################################################################
#
# Module : ContinuousGnowee.py
#
# Contains : Functions and methods for the Gnowee hybrid metaheuristic optimization
#            algorithm to perform non-linear constrainted and un-constrained optimization problems. 
#            This module only contains the routines for continuous optimization.  
#
# Author : James Bevins
#
# Last Modified: 16Aug16
#
#######################################################################################################

import ObjectiveFunctions as of
import SamplingMethods as sm
import Utilities as util
import numpy as np
import math as m
import copy as cp
import time

#---------------------------------------------------------------------------------------#
def Gnowee(func,lb,ub,S):
    """
    Main program for the HMO optimization. 
    
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
    pop=[]                     #List of parent objects
    
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
    
    #Initialize population with random initial solutions
    init=sm.Initial_Samples(lb,ub,S.s,S.p,S.d)
    for p in range(0,S.p,1):
        pop.append(util.Parent(5E20,init[p]))  
    if S.d:
        print 'Initial population:' 
        for i in range(S.p) :    
            print "Parent ",i,": Fitness=",pop[i].f,", Design=",pop[i].d
    
    #Calculate initial fitness values
    (pop,timeline)=util.Get_Best(obj,pop,pop,timeline,S)
    
    # Iterate until termination criterion met
    converge = False
    while timeline[-1].g <= S.gm and timeline[-1].e <= S.em and converge==False:
        
        # Global search using Levy flights and evaluate fitness
        children=Levy_Flight(pop,lb,ub,S)
        (pop,timeline)=util.Get_Best(obj,pop,children,timeline,S,random_replace=True)       
        
        # Perform regional search by mutating top subset of cuckoos
        (pop,timeline) = Elite_Crossover(obj,pop,lb,ub,timeline,S)
                                     
        # Local search using discovery and randomization
        children=Mutate(pop,lb,ub,S) 
        (pop,timeline)=util.Get_Best(obj,pop,children,timeline,S,random_replace=False)  
        
        # Test generational convergence
        if timeline[-1].g > S.sl:
            if timeline[-1].g > timeline[-2].g+S.sl:
                converge=True
                print "Generational Stall", timeline[-1].g,timeline[-2].g 
            elif timeline[-2].f==0:
                if S.of==0:
                    converge=True
                    print "Fitness Convergence"
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
def Levy_Flight(pop,lb,ub,S):
    """
    Generate new children using Levy flights
   
    Parameters
    ==========
    pop : list of Parent objects
        The current parents representing system designs
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
    children : list of parent objects
        The proposed children representing new system designs
    """
    
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Levy_Flight function.'
    assert len(lb)==len(pop[0].d), 'Bounds and pop have different #s of design variables in Levy_Flight function.'   
    
    children=[] # Local copy of children generated
            
    # Determine step size using Levy Flight
    step=sm.Levy(len(pop[0].d),len(pop),alpha=S.a,gamma=S.g,n=S.n,debug=S.d) 
    if S.d:
        print "\nIn function Levy_Flight,"
        print "The steps are:", step
        print S.a,S.g,S.n,S.ss/S.sf
    
    # Perform global search from fl*p parents
    used=[]
    for i in range(int(S.fl*S.p)):
        k=int(np.random.rand()*S.p)
        while k in used:
            k=int(np.random.rand()*S.p)
        used.append(k)
        children.append(cp.deepcopy(pop[k])) 
        
        #Calculate Levy flight
        stepsize=1.0/S.sf*step[k,:] 
        children[i].d=children[i].d+stepsize 
        if S.d:
            print "The old randomly chosen parent is", pop[i].d
            print "The step sizes are: ", stepsize
            print "The proposed #",i,"child, pre-boundary application, is: ",children[i].d
        
        #Build child applying variable boundaries 
        children[i].d=util.Rejection_Bounds(pop[k].d,children[i].d,stepsize,lb,ub,S,debug=S.d) 
        if S.d:
            print "The proposed #",i,"child, post-boundary application, is: ",children[i].d

    if S.d:
        print "The new proposed children are: ",
        for i in range(len(children)) : 
            print children[i].d
        
    return children

#---------------------------------------------------------------------------------------#               
def Mutate(pop,lb,ub,S):
    """
    Generate new children by adandoning pa parents
   
    Parameters
    ==========
    pop : list of parent objects
        The current parents representing system designs
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
    children : list of parent objects
        The children representing new system designs
    """
    from scipy.stats import rankdata
    
    assert len(pop[0].d)==len(lb), 'Pop and best have different #s of design variables in Abandon function.'
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Abandon function.'
    assert S.fd>=0 and S.fd <=1, 'The probability that a pop is discovered must exist on (0,1]'
    
    children=[]
            
    #Discover (1-fd); K is a status vector to see if discovered
    K=np.random.rand(len(pop),len(pop[0].d))>S.fd
        
    #Bias the discovery to the worst fitness solutions
    childn1=cp.copy(np.random.permutation(pop))
    childn2=cp.copy(np.random.permutation(pop)) 
        
    #New solution by biased/selective random walks
    r=np.random.rand()
    for j in range(0,len(pop),1):
        n=np.array((childn1[j].d-childn2[j].d))
        step_size=r*n
        tmp=pop[j].d+step_size*K[j,:]
        children.append(util.Parent(pop[j].f,util.Simple_Bounds(tmp,lb,ub,debug=S.d)))
        
    if S.d:
        print "\nAt end of function Abandon:"
        print "\nThe initial population was"
        for i in range(len(pop)):
            print "The parent #",i,"is ", pop[i].d, pop[i].f
        print "The status vector is :", K
        print "The step size is :", step_size
        for i in range(len(childn1)):
            print "The first permutation is ", childn1[i].f,childn1[i].d
        for i in range(len(childn2)):
            print "The second permutation is ", childn2[i].f,childn2[i].d
        for i in range(len(children)):
            print "The new child #",i,"is ", children[i].d
    return children      

#---------------------------------------------------------------------------------------#
def Elite_Crossover(func,pop,lb,ub,timeline,S):  
    """
    Generate new children using distance based crossover strategies on the top parent.  
    Ideas adapted from Walton "Modified Cuckoo Search: A New Gradient Free Optimisation Algorithm" 
    and Storn "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces"
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    pop : list of parent objects
        The current parents representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    timeline : list
        Storage list for best design event objects vs generation for new optimal designs 
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
   
    Returns
    =======
    pop : list of parent objects
        The new population of parents representing new system designs
    timeline : list
        Storage list for best design event objects vs generation for new optimal designs 
    """
    
    assert 0<=S.fe<=1, 'fe must be between 0 and 1 inclusive in Elite_Crossover function.'
    
    # Initialize variables
    golden_ratio=(1.+m.sqrt(5))/2.  # Used to bias distance based mutation strategies
    feval=0
    dx=np.zeros_like(pop[0].d)
            
    # Crossover top parent with an elite parent to speed local convergence
    top=[]
    top.append(cp.deepcopy(pop[0]))
    r=int(np.random.rand()*S.p*S.fe)
    if r==0:
        r=1
    top.append(cp.deepcopy(pop[r]))
    dx=abs(top[0].d-top[1].d)/golden_ratio
    top[0].d=top[1].d+dx
    top[0].d=util.Simple_Bounds(top[0].d,lb,ub,debug=False,change_count=0)

    # Evaluate fitness of child and compare to parent to see if improvement
    (fnew,gnew)=func(top[0].d)
    feval+=1
    if fnew < pop[r].f:
        pop[r]=cp.deepcopy(top[0])
        
    #Update timeline with new fitness values
    timeline[-1].e+=feval
                    
    if S.d:
        print "\nAt end of Elite_Crossover, the new population is: "
        for i in range(0,len(pop),1): 
            print "New parent #", i,":",pop[i].d 
        
    return pop, timeline
