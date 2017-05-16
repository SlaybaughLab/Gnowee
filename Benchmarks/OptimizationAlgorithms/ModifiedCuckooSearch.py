#######################################################################################################
#
# Module : ModifiedCuckooSearch.py
#
# Contains : Functions and methods for a Modified Cuckoo Search Algorithm developed by Walton.
#            to perform non-linear constrainted optimization problems.  The original  
#            algorithm is given in "Modified cuckoo search: A new gradient free optimisation algorithm". 
#            Modifications include:
#            1) Convergene criteria
#            2) Data Structure
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

#from pytest import set_trace

#---------------------------------------------------------------------------------------#
def MCS(func,lb,ub,S):
    """
    Perform a modified cuckoo search optimization (MCS) based on algorithm in 
    "Modified cuckoo search: A new gradient free optimisation algorithm", but modified 
    
    1) Convergene criteria
    
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
    timeline=[]                #List of event objects
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
    
    # Initialize nests with random initial solutions
    init=sm.Initial_Samples(lb,ub,S.s,n=S.c)
    for p in range(0,S.c,1):
        nests.append(csl.Nest(5E20,init[p]))  
    if S.d:
        print 'Initial nests:' 
        for i in range(S.c) :    
            print "Nest ",i,": Fitness=",nests[i].f,", Design=",nests[i].d 
    
    # Calculate initial fitness values
    (nests,timeline)=csl.Get_Best_Nest(obj,nests,nests,timeline,S)
    
    # Iterate until termination criterion met
    converge = False
    while timeline[-1].g <= S.gm and timeline[-1].e <= S.em and converge==False:
        
        # Decrease number of nests, 1 per generation until only 10 are left
        if len(nests) > 10:
            del nests[-1]
        
        # Global search using discovery
        (nests,timeline)=Discard_Nests(obj,nests,lb,ub,timeline[-1].g,S,timeline) 
        
        # Local search using Levy flights and evaluate fitness
        new_nests=Top_Nests(nests,lb,ub,timeline[-1].g,S)
        (nests,timeline)=csl.Get_Best_Nest(obj,nests,new_nests,timeline,S,random_replace=True)
        
        # Reset timeline
        timeline[-1].g=timeline[-1].g-1    
        
        # Global search using discovery and randomization
        new_nests=Empty_Nests(nests,lb,ub,timeline[-1].g,S)
        (nests,timeline)=csl.Get_Best_Nest(obj,nests,new_nests,timeline,S,random_replace=False)          
        
        # Test generational convergence
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
def Top_Nests(nests,lb,ub,gen,S):  
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
    gen : integer
        Current generation number
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
    assert 0<=S.pt<=1, 'pt must be between 0 and 1 inclusive in Get_Nests function.'
    
    tmp=[]  # Local copy of nest that is modified
    alpha=S.ss/S.sf/m.sqrt(gen)*(ub-lb)
    golden_ratio=(1.+m.sqrt(5))/2.  # Used to bias distance based mutation
    dx=np.zeros(len(nests[0].d))   # Distance between each variable for two selected solutions
            
    # Determine step size using Levy Flight
    step=sm.Levy(len(nests[0].d),int(S.pt*len(nests)),S.a,S.g,S.n)
    if S.d:
        print "\nAt beginning of Top_Nests: "        
        print "The steps are:", step
        print "The old nests are: "
        for i in range(0,len(nests),1) : 
            print "Old nest #", i,":",nests[i].d
        print "\n\nIn function Top_Nests,"
    
    # Perform global search from pc nests
    for i in range(int(S.pt*len(nests))):
        #Make a local copy of current nest        
        tmp.append(cp.copy(nests[i]))
        
        # Select random nest from within top nests to mutate with
        j=int(r.random()*S.pt*len(nests))
        
        if i==j:
            # Caclulate new nest position using Levy flights
            stepsize=alpha*step[i,:]*(tmp[i].d-nests[0].d)   
            if S.d:
                print "\nThe random nest is the same as nest #", i, "with design of ", tmp[i].d
                print "The step sizes are: ", stepsize
            tmp[i].d=tmp[i].d+stepsize
        
            #Built nest applying variable boundaries 
            if S.d:
                print "The proposed #",i,"nest, pre-boundary application, is: ",tmp[i].d
            tmp[i].d=csl.Rejection_Bounds(nests[i].d,tmp[i].d,lb,ub,debug=False)#S.d) 
            if S.d:
                print "The proposed #",i,"nest, post-boundary application, is: ",tmp[i].d

        else:
            dx=abs(tmp[i].d - nests[j].d)/golden_ratio
            if S.d:
                print "\nBefore distance mutation tmp[", i, "]= ", tmp[i].d
                print "The random nest is #", j, "with design of ", nests[j].d
            if i<j:
                tmp[i].d=nests[j].d+dx
            else:
                tmp[i].d=tmp[i].d+dx
            if S.d:
                print "After distance mutation tmp[", i, "]= ", tmp[i].d
                    
    if S.d:
        print "\nAt end of Top_Nests, the the old nests are: "
        for i in range(0,len(nests),1): 
            print "Old nest #", i,":",nests[i].d 
        print "\nAt end of Top_Nests, the new proposed nests are: "
        for i in range(0,len(tmp),1): 
            print "New nest #", i,":",tmp[i].d 
        print "The new proposed nests are: "
        for i in range(0,len(tmp),1): 
            print tmp[i].d
        
    return tmp
            
#---------------------------------------------------------------------------------------#               
def Empty_Nests(nests,lb,ub,gen,S):
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
    gen : integer
        Current generation number
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
    
    tmp=[]
            
    # Determine step size using Levy Flight
    step=sm.Levy(len(nests[0].d),len(nests),S.a,S.c,S.n)   
        
    #Discover pa nests; K is a status vector to see if discovered
    K=np.random.rand(len(nests),len(nests[0].d))>S.pa
        
    #Bias the discovery to the worst fitness solutions
    nestn1=cp.copy(np.random.permutation(nests))
    nestn2=cp.copy(np.random.permutation(nests))     
        
    #New solution by biased/selective random walks
    alpha=S.a/m.sqrt(gen)
    for j in range(0,len(nests),1):
        #Make a local copy of current nest        
        tmp.append(cp.copy(nests[j]))
        tmp[j].d=tmp[j].d+alpha*step[j,:]*K[j,:]*(tmp[j].d-nests[0].d) 
        tmp[j].d=csl.Simple_Bounds(tmp[j].d,lb,ub,debug=S.d)
        
    if S.d:
        print "\nAt end of function Empty_Nests:"
        print "The status vector is :", K
        print "The step size is :", step
        print "K * step size is :", K*step
        
        print "The first permutation is "
        for i in range(0,len(nestn1),1):
            print nestn1[i].f,nestn1[i].d
        
        print "The second permutation is "
        for i in range(0,len(nestn2),1):
            print nestn2[i].f,nestn2[i].d
            
        for i in range(0,len(tmp),1):
            print "The new nests #",i,"is ", tmp[i].d
            print len(tmp)
    return tmp           

#---------------------------------------------------------------------------------------#               
def Discard_Nests(func,nests,lb,ub,gen,S,timeline):
    """
    Generate new nests by emptying pa nests
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    nests : list of nest objects
        The current nests representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    gen : integer
        Current generation number
    S : Object    
        An object representing the settings for the optimization algorithm
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
   
    Optional
    ========   
   
    Returns
    =======
    nests : list of nest objects
        The new nests representing system designs
    timeline : list of history objects
        The updated history of the optimization process containing best design, fitness, 
        generation, and function evaluations
    """
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    assert len(nests[0].d)==len(lb), 'Nest and best have different #s of design variables in Empty_Nests function.'
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Empty_Nests function.'
    assert S.pa>0 and S.pa <=1, 'The probability that a nest is discovered must exist on (0,1]'
    
    feval=0
    
    # Determine step size using Levy Flight
    alpha=S.ss/S.sf/m.sqrt(gen)*(ub-lb)
    step=sm.Levy(len(nests[0].d),len(nests),S.a,S.c,S.n)       
        
    #New solution by biased/selective random walks
    for j in range(len(nests)-1,int(len(nests)*S.pt),-1):
        #Make a local copy of current nest        
        tmp=cp.copy(nests[j].d)
        tmp=tmp+alpha*step[j,:] 
        tmp=csl.Rejection_Bounds(nests[j].d,tmp,lb,ub,debug=False)#S.d)
        (fnew,gnew)=func(tmp)
        feval += 1
        nests[j].d=tmp
        nests[j].f=fnew
        
    #Sort the nests
    nests.sort(key=lambda x: x.f)
        
    #Store history on timeline if new optimal design found
    if nests[0].f<timeline[-1].f and (timeline[-1].f-nests[0].f)/nests[0].f > S.ct:
        timeline.append(csl.Event(timeline[-1].g,timeline[-1].e+feval,nests[0].f,nests[0].d))
    else:
        timeline[-1].e+=feval
        
    if S.d:
        print "\nAt end of function Discard_Nests:"
        print "The step size is :", step            
        for i in range(0,len(nests),1):
            print "The new nests #",i,"has fitness=",nests[i].f," and is ", nests[i].d
    return nests,timeline     