#######################################################################################################
#
# Module : ZhouDiscreteCuckooSearch.py
#
# Contains : Functions and methods for a Discrete Cuckoo Search Algorithm developed by Zhou et al.
#            to perform combinatorial (TSP) optimization problems.  The original  
#            algorithm is given in "A Novel Discrete Cuckoo Search Algorithm for Spherical Traveling 
#            Salesman Problem" and "A Discrete Cuckoo Search Algorithm for Travelling Salesman Problem."             
#
# Author : James Bevins
#
# Last Modified: 22Oct15
#
###############################################################################################################################

import ObjectiveFunctions as of
import SamplingMethods as sm
import CSLibrary as csl
import numpy as np
import random as r
import math as m
import copy as cp
import time
import heapq
from scipy.stats import rankdata

#from pytest import set_trace

#---------------------------------------------------------------------------------------#
def DCS(func,S,tsp):
    """
    Perform a discrete cuckoo search optimization (DCS) based on algorithm in 
    "Discrete Cuckoo Search Algorithm for the Travelling Salesman Problem," 
    
    Parameters
    ==========
    func : function
        The objective function to be minimized
    S : Object    
        An object representing the settings for the optimization algorithm
    tsp : Object
        An object containing the problem information
   
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
        
    # Check for objective function(s) 
    assert hasattr(func, '__call__'), 'Invalid function handle'
    obj = lambda x: func(x)    
    
    # Set seed to aid in debug mode
    if S.d:
        np.random.seed(42)
        r.seed(100)
    
    # Create Effectiveness Tracker
    eff_stats=Effectiveness_Tracker()
    
    # Generate S.c nests using Levy flight operation
    nests=Flight_Operation(nests,S.c,tsp,debug=S.d)

    # Calculate initial fitness values
    (nests,timeline)=csl.Get_Best_Nest(obj,nests,nests,timeline,S)
            
    # Improve starting nest using partial inversion
    (nests,timeline,eff_stats)=Partial_Inversion(obj,nests,S.c,timeline,eff_stats,debug=S.d)
    
    if S.d:
        print 'Initial nests:' 
        for i in range(S.c) :    
            print "Nest ",i,": Fitness=",nests[i].f,", Design=",nests[i].d
    
    # Iterate until termination criterion met
    converge = False
#    while timeline[-1].g <= S.gm and timeline[-1].e <= S.em and converge==False:
    while timeline[-1].e <= S.em and converge==False:
        
        # Generate new nests via Levy flight and New Nest Operator
        (nests,timeline,eff_stats)=New_Nests(obj,nests,timeline,S,eff_stats)  
        
        # Apply Study Operator
        (nests,timeline,eff_stats)=Study_Operator(obj,nests,timeline,S,eff_stats)
        
        # Apply A operator
        nests.sort(key=lambda x: x.f) 
        (nests,timeline,eff_stats)=A_Operator(func,nests,timeline,S,eff_stats)
        
        # Apply 3-opt
        (nests,timeline,eff_stats)=Three_Opt(func,nests,timeline,S,eff_stats)
        
        # Global search using discovery
        (nests,timeline,eff_stats)=Discard_Nests(obj,nests,S,timeline,tsp,eff_stats)        

        # Sort and remove worst nests
        nests.sort(key=lambda x: x.f)  
            
#        if nests[0].f<timeline[-1].f and (timeline[-1].f-nests[0].f)/nests[0].f > S.ct:
        timeline.append(csl.Event(timeline[-1].g+1,timeline[-1].e,nests[0].f,nests[0].d))
#        else:
#            timeline[-1].g+=1 
            
        if True:
            print "Generation =",timeline[-1].g,",fitness=",timeline[-1].f,", and fevals=",timeline[-1].e
            
        # Test generational convergence
#        if timeline[-1].g > S.sl:
#            if timeline[-1].g > timeline[-2].g+S.sl:
#                converge=True
#                print "Generational Stall", timeline[-1].g,timeline[-2].g 
#            elif timeline[-1].f ==timeline[(-1-S.sl)].f:
#                converge=True
#                print "Generational Stall", timeline[-1].g,timeline[(-1-S.sl)].g 
#            elif (timeline[-2].f-timeline[-1].f)/timeline[-2].f < S.ct:
#                if timeline[-1].g > timeline[-2].g+S.sl:
#                    converge=True
#                    print "Generational Convergence"
                
        # Test fitness convergence
#        if ((timeline[-1].f-S.of)/S.of) <= S.ot:
#            converge=True
#            print "Fitness Convergence" 
            
    #Determine execution time
    print "Program execution time was %f."  % (time.time() - start_time)
    
    # Print Effectiveness Diagnostics
    if True:
        print "\nThe # of Partial_Inversion evals was ",eff_stats.pie,"with a change rate of",\
            eff_stats.pic/float(eff_stats.pie)
        print "\nThe # of New_Nest evals was ",eff_stats.nne,"with a change rate of",eff_stats.nnc/float(eff_stats.nne) 
        print "\nThe # of Study_Operator evals was ",eff_stats.soe,"with a change rate of",eff_stats.soc/float(eff_stats.soe) 
        print "\nThe # of A_Operator evals was ",eff_stats.aoe,"with a change rate of",eff_stats.aoc/float(eff_stats.aoe)
        print "\nThe # of 3-opt evals was ",eff_stats.toe,"with a change rate of",eff_stats.toc/float(eff_stats.toe)
        print "\nThe # of Discard_Nests evals was ",eff_stats.dne,"with a change rate of",eff_stats.dnc/float(eff_stats.dne)

    if S.d:
        print "\nAt end of optimization:"      
        for i in range(len(nests)):
            print "The new nests #",i,"has fitness=",nests[i].f," and is ", nests[i].d    
        
    return timeline
        
#---------------------------------------------------------------------------------------#
def New_Nests(func,nests,timeline,S,stats):  
    """
    Generate new nests using Levy flights
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    nests : list of nest objects
        The current nests representing system designs
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
    S : Object    
        An object representing the settings for the optimization algorithm
    stats : Object    
        An object that tracks the function eval and replacement statistics to evaulate effectiveness
   
    Optional
    ========   
   
    Returns
    =======
    nests : list of nest objects
        The proposed nests representing new system designs
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    # Initialize Variables
    feval=0
    n=len(nests)
    for i in range(0,n,1):
        nests.append(csl.Nest(nests[i].f,cp.copy(nests[i].d),nests[i].i))
        while nests[i+n].f>=nests[i].f:    
            dist=np.zeros(len(nests[i].d))   #Reset distance vector       
            # Calculate distance to each of the cities
            for j in range(0,len(nests[i].d),1):
                dist[j]=round(m.sqrt((nests[i+n].d[j][0]-nests[i+n].d[nests[i+n].i][0])**2 + \
                                     (nests[i+n].d[j][1]-nests[i+n].d[nests[i+n].i][1])**2))
                if j==nests[i+n].i:
                    dist[j]=1E99
                    
            #Based on Levy flights, determine next city to visit
            p=abs(sm.Levy(1))
            if p<1:
                ind=np.argmin(dist)  #Determine the index of the closest city
            elif p>=1:
                d=np.min(dist)*p   #Use Levy flight to determine maiumum distance searched
                ind=np.where(dist==np.max(dist[dist <= d]))[0][0] #Determine the index of the furthest city within d
            else:
                print "An invalid value of p was generated using levy flights."
             
            # If the closest city and randomly chosen city are non-adjacent, invert the cities btwn closest (inclused) and 
            # current city (not included)
            if ind>nests[i+n].i+1 or ind<nests[i+n].i-1:
                if ind<nests[i+n].i:
                    nests[i+n].d[ind+1:nests[i+n].i+1]=reversed(nests[i+n].d[ind+1:nests[i+n].i+1])
                    (nests[i+n].f,g)=func(nests[i+n].d)
                else: 
                    nests[i+n].d[nests[i+n].i:ind]=reversed(nests[i+n].d[nests[i+n].i:ind])
                    (nests[i+n].f,g)=func(nests[i+n].d)
                feval+=1    #Update function eval counter
                stats.nne+=1
                nests[i+n].i=ind   # For the ith nest, update the index of the most recently visited city
            else:
                nests[i+n].i=ind   # For the ith nest, update the index of the most recently visited city
                break
        if nests[i+n].f<nests[i].f:
            stats.nnc+=1

    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval            
    
    # Remove old nests
    del nests[n:len(nests)]     # Unclear if this is true to Zhou DCS algorithm
    
    if S.d:
        print "\nAt end of New_Nests, the new proposed nests are: "
        for i in range(0,len(nests),1): 
            print "New nest #", i,":",nests[i].f,nests[i].d
      
    return nests,timeline,stats

#---------------------------------------------------------------------------------------#
def Study_Operator(func,nests,timeline,S,stats):  
    """
    Generate new nests using Levy flights
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    nests : list of nest objects
        The current nests representing system designs
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
    S : Object    
        An object representing the settings for the optimization algorithm
    stats : Object    
        An object that tracks the function eval and replacement statistics to evaulate effectiveness
   
    Optional
    ========   
   
    Returns
    =======
    nests : list of nest objects
        The proposed nests representing new system designs
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """ 
        
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    feval=0
    for i in range(0,len(nests),1):
        # Initialize variables
        rand1=i
        rand2=int(np.random.rand()*len(nests))  #Randomly choose starting nest #2
        while rand2==rand1:
            rand2=int(np.random.rand()*len(nests))  #Randomly choose starting nest #2

        # Loop over every city in the nest
        for c1 in range(0,len(nests[rand1].d),1):
            d2=(contains_sublist(nests[rand2].d,nests[rand1].d[c1])+1)%len(nests[rand1].d)
            d1=(contains_sublist(nests[rand1].d,nests[rand2].d[d2]))
            c2=(contains_sublist(nests[rand2].d,nests[rand1].d[(d1+1)%len(nests[rand1].d)]))%len(nests[rand1].d)
            # Test permutation in first nest
            tmp=cp.copy(nests[rand1].d)
            if c1<d1:
                tmp[c1+1:d1+1]=reversed(tmp[c1+1:d1+1])
            else:
                tmp[d1:c1]=reversed(tmp[d1:c1])

            # If improved, save; else test permutation in second nest
            (ftmp,g)=func(tmp)
            feval+=1
            stats.soe+=1
            if ftmp<nests[rand1].f:
                nests[rand1].d=cp.copy(tmp)
                nests[rand1].f=ftmp
                stats.soc+=1
            else:    
                tmp=cp.copy(nests[rand2].d)
                if c2<d2:
                    tmp[c2:d2]=reversed(tmp[c2:d2])
                else:
                    tmp[d2+1:c2+1]=reversed(tmp[d2+1:c2+1])
                (ftmp,g)=func(tmp)
                feval+=1
                stats.soe+=1
                if ftmp<nests[rand2].f:
                    nests[rand2].d=cp.copy(tmp) 
                    nests[rand2].f=ftmp
                    stats.soc+=1
        
        
    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval 
    
    # Debug Statements
    if S.d:
        print "\nAt end of Study_Operator, the new proposed nests are: "
        for i in range(0,len(nests),1): 
            print "New nest #", i,":",nests[i].f,nests[i].d
            
    return nests,timeline,stats
    
#---------------------------------------------------------------------------------------#
def A_Operator(func,nests,timeline,S,stats):  
    """
    Generate new nests using Levy flights
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    nest : a nest object
        The current nests representing system designs
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
    S : Object    
        An object representing the settings for the optimization algorithm
    stats : Object    
        Object that tracks the function eval and replacement statistics to evaulate effectiveness
   
    Optional
    ========   
   
    Returns
    =======
    nests : nest objects
        The proposed nest representing new system design
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """ 
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    # Initialize variables
    feval=0
    for i in range(0,len(nests),1):    
        for a in range(0,len(nests[i].d),1):
            b=(a+1)%len(nests[i].d)  #City #2
            c=(a+2)%len(nests[i].d)    #City #3

            # Calculate distance to each of the cities
            dist=np.zeros(len(nests[i].d))   #Reset distance vector 
            for j in range(0,len(nests[i].d),1):
                dist[j]=round(m.sqrt((nests[i].d[j][0]-nests[i].d[b][0])**2 + (nests[i].d[j][1]-nests[i].d[b][1])**2))
                if j==b:
                    dist[j]=1E99

            #Based on Levy flights, determine next city to visit
            p=abs(sm.Levy(1))
            if p<1:
                d=np.argmin(dist)  #Determine the index of the closest city
            elif p>=1:
                tmp=np.min(dist)*p   #Use Levy flight to determine maxiumum distance searched
                d=np.where(dist==np.max(dist[dist <= tmp]))[0][0] #Determine the index of the furthest city within d
            else:
                 print "An invalid value of p was generated using levy flights."

            e=(d+1)%len(nests[i].d)    #City #5

            # Construct partial tour and determine length; test for coincidences to construct segments
            # Handle the special case
            if e==a:
                old_dist=round(m.sqrt((nests[i].d[b][0]-nests[i].d[c][0])**2 + (nests[i].d[b][1]-nests[i].d[c][1])**2) + \
                              m.sqrt((nests[i].d[a][0]-nests[i].d[d][0])**2 + (nests[i].d[a][1]-nests[i].d[d][1])**2) )
                new_dist=round(m.sqrt((nests[i].d[a][0]-nests[i].d[c][0])**2 + (nests[i].d[a][1]-nests[i].d[c][1])**2) + \
                              m.sqrt((nests[i].d[b][0]-nests[i].d[d][0])**2 + (nests[i].d[b][1]-nests[i].d[d][1])**2) )
                feval+=1  
                stats.aoe+=1

                # Compare path lengths, change nest if new path is shorter
                if new_dist<old_dist:
                    hold=nests[i].d[b]
                    del nests[i].d[b]
                    nests[i].d.insert(a,hold)    

                    # Update the function eval counter in the timeline
                    (nests[i].f,g)=func(nests[i].d)
                    feval+=1  
                    stats.aoe+=1
                    stats.aoc+=1

            # Ignore all of the cases where the distance is the same       
            elif d!=a and d!=c and e!=b:
                old_dist=round(m.sqrt((nests[i].d[a][0]-nests[i].d[b][0])**2 + (nests[i].d[a][1]-nests[i].d[b][1])**2) + \
                              m.sqrt((nests[i].d[b][0]-nests[i].d[c][0])**2 + (nests[i].d[b][1]-nests[i].d[c][1])**2) + \
                              m.sqrt((nests[i].d[d][0]-nests[i].d[e][0])**2 + (nests[i].d[d][1]-nests[i].d[e][1])**2) )
                new_dist=round(m.sqrt((nests[i].d[a][0]-nests[i].d[c][0])**2 + (nests[i].d[a][1]-nests[i].d[c][1])**2) + \
                              m.sqrt((nests[i].d[d][0]-nests[i].d[b][0])**2 + (nests[i].d[d][1]-nests[i].d[b][1])**2) + \
                              m.sqrt((nests[i].d[b][0]-nests[i].d[e][0])**2 + (nests[i].d[b][1]-nests[i].d[e][1])**2) )
                feval+=1  
                stats.aoe+=1

                # Compare path lengths, change nest if new path is shorter
                if new_dist<old_dist:
                    hold=nests[i].d[b]
                    del nests[i].d[b]
                    if b<e:
                        nests[i].d.insert(d,hold)
                    else:
                        nests[i].d.insert(e,hold)

                    # Update the function eval counter in the timeline
                    (nests[i].f,g)=func(nests[i].d)
                    feval+=1
                    stats.aoe+=1
                    stats.aoc+=1
    
    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval 
    
    # Debug Statements
    if S.d:
        print "\nAt end of Study_Operator, the new proposed nests are: "
        print "New nest :",nests[i].f,nests[i].d
          
    return nests,timeline,stats

#---------------------------------------------------------------------------------------#
def Three_Opt(func,nests,timeline,S,stats):  
    """
    Generate new nests using Levy flights
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    nest : a nest object
        The current nests representing system designs
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
   
    Returns
    =======
    nests : nest object
        The proposed nest representing new system design
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    # Initialize variables
    feval=0
    for i in range(0,len(nests),1): 
        tmp=[]   # Make a local copy of current nest designs

        # Generate 3 random nodes
        breaks=np.sort(np.random.rand(3)*len(nests[i].d)//1+1)%len(nests[i].d)
        #Ensure that 3 different nodes are selected
        while breaks[1]==breaks[0] or breaks[1]==breaks[2]:
            #breaks[1]=np.sort(np.random.rand()*len(nests[i].d)//1+1)%len(nests[i].d)    
            breaks[1]=(np.random.rand()*len(nests[i].d)//1+1)%len(nests[i].d)
        while breaks[2]==breaks[0]:
            #breaks[2]=np.sort(np.random.rand()*len(nests[i].d)//1+1)%len(nests[i].d)
            breaks[2]=(np.random.rand()*len(nests[i].d)//1+1)%len(nests[i].d)
        breaks=np.sort(breaks)   

        # Make reconnections first way
        tmp[0:int(breaks[0])]=nests[i].d[0:int(breaks[0])]
        tmp[len(tmp):int(len(tmp)+breaks[2]-breaks[1])]=nests[i].d[int(breaks[1]):int(breaks[2])]
        tmp[len(tmp):int(len(tmp)+breaks[1]-breaks[0])]=nests[i].d[int(breaks[0]):int(breaks[1])]
        tmp[len(tmp):int(len(tmp)+breaks[2]-len(nests[i].d))]=nests[i].d[int(breaks[2]):len(nests[i].d)]

        # Test new route against old; update if better
        (ftmp,g)=func(tmp)
        feval+=1
        stats.toe+=1
        if ftmp<nests[i].f:
            nests[i].d=cp.copy(tmp) 
            nests[i].f=ftmp
            stats.toc+=1

        # Make reconnections second way
        tmp=[]   # Reset local copy of current nest designs
        tmp[0:int(breaks[0])]=nests[i].d[0:int(breaks[0])]
        tmp[len(tmp):int(len(tmp)+breaks[1]-breaks[0])]=reversed(nests[i].d[int(breaks[0]):int(breaks[1])])
        tmp[len(tmp):int(len(tmp)+breaks[2]-breaks[1])]=reversed(nests[i].d[int(breaks[1]):int(breaks[2])])
        tmp[len(tmp):int(len(tmp)+breaks[2]-len(nests[i].d))]=nests[i].d[int(breaks[2]):len(nests[i].d)]

        # Test new route against old; update if better
        (ftmp,g)=func(tmp)
        feval+=1
        stats.toe+=1
        if ftmp<nests[i].f:
            nests[i].d=cp.copy(tmp) 
            nests[i].f=ftmp
            stats.toc+=1
        
    # Update the function eval counter in the timeline
    (nests[i].f,g)=func(nests[i].d)
    feval+=1
    timeline[-1].e=timeline[-1].e+feval 
    
    # Debug Statements
    if S.d:
        print "\nAt end of Study_Operator, the new proposed nests are: "
        print "New nest :",nests[i].f,nests[i].d
    
    return nests,timeline,stats

#---------------------------------------------------------------------------------------#               
def Discard_Nests(func,nests,S,timeline,prob_in,stats):
    """
    Generate new nests by emptying pa nests
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    nests : list of nest objects
        The current nests representing system designs
    S : Object    
        An object representing the settings for the optimization algorithm
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
    prob_in : Object
        An object containing the problem information
    stats : Object    
        Object that tracks the function eval and replacement statistics to evaulate effectiveness
   
    Optional
    ========   
   
    Returns
    =======
    nests : list of nest objects
        The new nests representing system designs
    timeline : list of history objects
        The updated history of the optimization process containing best design, fitness, 
        generation, and function evaluations
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    assert S.pa>0 and S.pa <=1, 'The probability that a nest is discovered must exist on (0,1]'
    
    #Initialize variables
    feval=0
    
    #Determine weights based on fitness    
    weights=csl.WeightedRandomGenerator(rankdata(range(0,len(nests),1)))
    
    for i in range(0,int(S.pa*len(nests)),1):
        tmp=[]   #Working copy of chosen nest
        discard=next(weights)    #Linearly weighted random nest to discard

        # Perform Levy flights to get initial nest
        tmp=Flight_Operation(tmp,1,prob_in,debug=S.d)
        tmp[0].f=func(tmp[0].d)
        feval+=1
        stats.dne+=1

        # Perform partial inversion to improve nest
        (tmp,timeline,stats)=Partial_Inversion(func,tmp,1,timeline,stats,debug=S.d)

        # Check to see if new nest is better
        if tmp[0].f<nests[discard].f:
            stats.dnc+=1

        # Update discarded nest
        nests[discard]=cp.deepcopy(tmp[0])
    
    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval 
    
    if S.d:
        print 'After discard_nests:' 
        for i in range(n) :    
            print "Nest ",i,": Fitness=",nests[i].f,", Design=",nests[i].d
          
    return nests,timeline,stats  

#---------------------------------------------------------------------------------------#               
def Flight_Operation(nests,n,prob_in,debug=False):
    """
    Generate new nests by emptying pa nests
   
    Parameters
    ==========
    nests : list of nest objects
        The current nests representing system designs
    n : integer    
        Number of nests to generate
    prob_in : Object
        An object containing the problem information
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)  
   
    Returns
    =======
    nests : list of nest objects
        The new nests representing system designs
    """
 
    # Initialize nests with random initial solution based on Levy flights
    for i in range(0,n,1):
        tmp=cp.copy(prob_in.n)   #Local copy of the cities
        rand=int(np.random.rand()*len(tmp))  #Randomly choose starting nest
        nests.append(csl.Nest(5E20,[]))  #Create an instance of Nest object
        nests[i].d.append(list(tmp[rand])) #Final route -> append starting nest
        tmp=np.delete(tmp,rand,0)   #Delete the chosen nest from the temporary copy so it is not visited again
        
        # Loop over the list of cities until all have been visited
        while len(tmp) != 0:
            dist=np.zeros(len(tmp))   #Reset distance vector
            # Calculate distance to each of the remaining cities
            for j in range(0,len(tmp),1):
                dist[j]=round(m.sqrt((tmp[j][0]-nests[i].d[-1][0])**2 + (tmp[j][1]-nests[i].d[-1][1])**2))
            #Based on Levy flights, determine next city to visit
            p=abs(sm.Levy(1))
            if p<1:
                nests[i].d.append(list(tmp[np.argmin(dist)]))
                tmp=np.delete(tmp,np.argmin(dist),0)
            elif p>=1:
                d=np.min(dist)*p
                nests[i].d.append(tmp[np.where(dist==np.max(dist[dist <= d]))[0]][0].reshape(2).tolist())
                tmp=np.delete(tmp,np.where(dist==np.max(dist[dist <= d]))[0][0],0)
            else:
                print "An invalid value of p was generated using levy flights."
    
    if debug:
        print 'After flight operations:' 
        for i in range(n) :    
            print "Nest ",i,": Fitness=",nests[i].f,", Design=",nests[i].d
            
    return nests   

#---------------------------------------------------------------------------------------#               
def Partial_Inversion(func,nests,n,timeline,stats,debug=False):
    """
    Generate new nests by emptying pa nests
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    nests : list of nest objects
        The current nests representing system designs
    n : integer    
        Number of nests to generate
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)  
   
    Returns
    =======
    nests : list of nest objects
        The new nests representing system designs
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """  
    
    feval=0
    for i in range(0,n,1):
        tmp=cp.deepcopy(nests[i])   #Local copy of the nest object
        
        # Randomly establish starting point
        rand=int(np.random.rand()*len(tmp.d))  
        order=[]
        order[0:(len(tmp.d)-rand)]=range(rand,len(tmp.d),1)
        order[(len(tmp.d)-rand):len(tmp.d)]=range(0,rand,1)
        nests[i].i=order[-1]
        
        # Visit each city in the order specified
        while len(order)!=0: 
            dist=np.zeros(len(tmp.d))   #Reset distance vector
            tmp=cp.deepcopy(nests[i])   #Reset the local copy of the nest object (Not original Zhou - I don't think)
            
            # Calculate distance to each of the cities
            for j in range(0,len(tmp.d),1):
                dist[j]=round(m.sqrt((tmp.d[j][0]-tmp.d[order[0]][0])**2 + (tmp.d[j][1]-tmp.d[order[0]][1])**2))
                if j==order[0]:
                    dist[j]=1E99
            
            # If the closest city and randomly chosen city are non-adjacent, invert the cities btwn closest (inclused) and 
            #furthest city (not included)
            imin=np.argmin(dist)  #Determine the index of the closest city
            nests[i].i=imin   # For the ith nest, update the index of the most recently visited city
            if imin>order[0]+1 or imin<order[0]-1:
                imax=np.where(dist==heapq.nlargest(2, dist)[1])[0][0]  #Determine the index of the furthest city
                if imin<imax:
                    tmp.d[imin:imax]=reversed(tmp.d[imin:imax])
                    (tmp.f,g)=func(tmp.d)
                else: 
                    tmp.d[imax+1:imin+1]=reversed(tmp.d[imax+1:imin+1])
                    (tmp.f,g)=func(tmp.d)
                feval+=1   # Update the number of function evals performed
                stats.pie+=1
                
                # Test inversion to see if lower fitness; if so save the solution
                if tmp.f<nests[i].f:
                    nests[i]=cp.deepcopy(tmp)
                    stats.pic+=1

            # Advance to the nest city
            del order[0]
            
    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval 
    
    if debug:
        print 'After partial inversion:' 
        for i in range(n) :    
            print "Nest ",i,": Fitness=",nests[i].f,", Design=",nests[i].d  
            
    return nests,timeline,stats
    
#---------------------------------------------------------------------------------------#               
def Read_TSP(filename,debug=False):
    """
    Generate new nests by emptying pa nests
   
    Parameters
    ==========
    filename : string
        Path and filename of the tsp problem
   
    Optional
    ======== 
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)  
   
    Returns
    =======
    prob : TSP object
        An obect containing the TSP problem parameters
    """
    
    # Initialize variables
    header=True
    prob=TSP()
    prob.n=[]
    print filename
    
    
    # Open file, read line by line, and store tsp problem parameters to the TSP object
    with open(filename,'r') as f:
        line=f.readline()
        key, value = line.split(":")
        prob.name=value.strip() 
        for line in f:
            if header==True:
                key, value = line.split(":")
                if key.strip()=='DIMENSION':
                     prob.d=int(value.strip())
                if key.strip()=='EDGE_WEIGHT_TYPE':
                    header=False
                    next(f)
            elif header==False:
                if line.strip() != 'EOF':
                    split_list=line.split()
                    if len(split_list) != 3:
                        raise ValueError("Line {}: '{}' has {} spaces, expected 1" \
                        .format(line, line.rstrip(), len(split_list) - 1))
                    else:
                        node,x,y = split_list
                        prob.n.append([float(x.strip()),float(y.strip())]) 
    
    # Test that the file closed
    assert f.closed==True, "File did not close properly."

    # Debug statements
    if debug==True:
        print "The problem is :", prob.name
        print "It has %d nodes, which are:" %prob.d
        print prob.n

    return prob

#---------------------------------------------------------------------------------------#    
class TSP:
    """
    Creates a tsp object 
   
    Attributes
    ==========
    name : string
        The name of the TSPLIB problem
    dimension : integer
        The number of nodes in the problem
    nodes : list
         The coorinate pairs for each node
    Returns
    =======
    None
    """
        
    def __init__(self,name='',dimension=1,nodes=[]):
        self.name=name
        self.d=dimension
        self.n=nodes
        
#---------------------------------------------------------------------------------------# 
def contains_sublist(lst, sublst):
    """
    Find index of sublist, if it exists
    
    Parameters
    ==========
    lst : list
        The list in which to search for sublst
    sublst : list
        The list to search for 
   
    Optional
    ========
   
    Returns
    =======
    i : integer
        Location of sublst in lst
    """
    for i in range(0,len(lst),1):
        if sublst == lst[i]:
            return i
        
#---------------------------------------------------------------------------------------#    
class Effectiveness_Tracker:
    """
    Creates a effectiveness tracker object to determine which methods produce the greatest result
   
    Attributes
    ==========
    partial_inversion_changes : integer
        The number of time the partial inversion method updated the current solution
    partial_inversion_fevals : integer
        The number of time the partial inversion method performed a function eval
    new_nests_changes : integer
        The number of time the new_nests method updated the current solution
    new_nests_fevals : integer
        The number of time the new_nests method performed a function eval
    study_operator_changes : integer
        The number of time the learning_operator method updated the current solution
    study_operator_fevals : integer
        The number of time the learning_operator method performed a function eval
    a_operator_changes : integer
        The number of time the a_operator method updated the current solution
    a_operator_fevals : integer
        The number of time the a_operator method performed a function eval
    three_opt_changes : integer
        The number of time the three_opt method updated the current solution
    three_opt_fevals : integer
        The number of time the three_opt method performed a function eval
    discard_nests_changes : integer
        The number of time the discard_nests method updated the current solution
    discard_nests_fevals : integer
        The number of time the discard_nests method performed a function eval
    Returns
    =======
    None
    """
        
    def __init__(self,partial_inversion_changes=0,partial_inversion_fevals=0,new_nests_changes=0,new_nests_fevals=0, \
                 study_operator_changes=0,study_operator_fevals=0,a_operator_changes=0,a_operator_fevals=0,\
                 three_opt_changes=0,three_opt_fevals=0,discard_nests_changes=0,discard_nests_fevals=0):
        self.pic=partial_inversion_changes
        self.pie=partial_inversion_fevals
        self.nnc=new_nests_changes
        self.nne=new_nests_fevals
        self.soc=study_operator_changes
        self.soe=study_operator_fevals
        self.aoc=a_operator_changes
        self.aoe=a_operator_fevals
        self.toc=three_opt_changes
        self.toe=three_opt_fevals
        self.dnc=discard_nests_changes
        self.dne=discard_nests_fevals