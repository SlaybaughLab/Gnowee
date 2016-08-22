#######################################################################################################
#
# Module : DiscreteGnowee.py
#
# Contains : Functions and methods for the Gnowee hybrid metaheuristic optimization algorithm.  This implementations
#            only contains the method for for dicrete problems.  
#
# Author : James Bevins
#
# Last Modified: 16Aug16
#
###############################################################################################################################

import ObjectiveFunctions as of
import SamplingMethods as sm
import Utilities as util
import numpy as np
import math as m
import copy as cp
import time
import heapq
from scipy.stats import rankdata

#---------------------------------------------------------------------------------------#
def Gnowee(func,S,tsp):
    """
    Perform a Hybrid Metaheuristic Optimization search
    
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
    pop=[]                     #List of parent objects
        
    # Check for objective function(s) 
    assert hasattr(func, '__call__'), 'Invalid function handle'
    obj = lambda x: func(x)    
    
    # Set seed to aid in debug mode
    if S.d:
        np.random.seed(42)
    
    # Create Effectiveness Tracker
    eff_stats=Effectiveness_Tracker()
    
    # Generate S.p parents using Levy flight operation
    pop=Flight_Operation(pop,S.p,tsp,debug=S.d)

    # Calculate initial fitness values
    (pop,timeline)=util.Get_Best(obj,pop,pop,timeline,S)
            
    # Improve starting population using partial inversion
    (pop,timeline,eff_stats)=Partial_Inversion(obj,pop,S.p,timeline,eff_stats,debug=S.d)
    
    if S.d:
        print 'Initial pop:' 
        for i in range(S.p) :    
            print "Parent ",i,": Fitness=",pop[i].f,", Design=",pop[i].d
    
    # Iterate until termination criterion met
    converge = False
    while timeline[-1].g <= S.gm and timeline[-1].e <= S.em and converge==False:
        
        # Generate new children via Levy flight 
        (pop,timeline,eff_stats)=Levy_Flight(obj,pop,timeline,S,eff_stats) 
        
        # Apply Crossover
        (pop,timeline,eff_stats)=Crossover(obj,pop,timeline,S,eff_stats)
        
        # Apply 2-opt
        pop.sort(key=lambda x: x.f) 
        (pop,timeline,eff_stats)=Two_opt(func,pop,timeline,S,eff_stats)
        
        # Apply 3-opt
        (pop,timeline,eff_stats)=Three_Opt(func,pop,timeline,S,eff_stats)  
             
        pop.sort(key=lambda x: x.f)
        timeline.append(util.Event(timeline[-1].g+1,timeline[-1].e,pop[0].f,pop[0].d))
            
        if S.d:
            print "Generation =",timeline[-1].g,",fitness=",timeline[-1].f,", and fevals=",timeline[-1].e
            
        # Test generational convergence
        if timeline[-1].g > S.sl:
            if timeline[-1].g > timeline[-2].g+S.sl:
                converge=True
                print "Generational Stall", timeline[-1].g,timeline[-2].g 
            elif timeline[-1].f ==timeline[(-1-S.sl)].f:
                converge=True
                print "Generational Stall", timeline[-1].g,timeline[(-1-S.sl)].g 
            elif (timeline[-2].f-timeline[-1].f)/timeline[-2].f < S.ct:
                if timeline[-1].g > timeline[-2].g+S.sl:
                    converge=True
                    print "Generational Convergence"
                
        # Test fitness convergence
        if ((timeline[-1].f-S.of)/S.of) <= S.ot:
            converge=True
            print "Fitness Convergence" 
            
    #Determine execution time
    print "Program execution time was %f."  % (time.time() - start_time)
    
    # Print Effectiveness Diagnostics
    if S.d:
        print "\nThe # of Partial_Inversion evals was ",eff_stats.pie,"with a change rate of",\
            eff_stats.pic/float(eff_stats.pie)
        print "\nThe # of Levy_Flight evals was ",eff_stats.lfe,"with a change rate of",eff_stats.lfc/float(eff_stats.lfe) 
        print "\nThe # of Crossover evals was ",eff_stats.ce,"with a change rate of",eff_stats.cc/float(eff_stats.ce) 
        print "\nThe # of A_Operator evals was ",eff_stats.aoe,"with a change rate of",eff_stats.aoc/float(eff_stats.aoe)
        print "\nThe # of 3-opt evals was ",eff_stats.toe,"with a change rate of",eff_stats.toc/float(eff_stats.toe)

    if S.d:
        print "\nAt end of optimization:"      
        for i in range(len(pop)):
            print "The new parent #",i,"has fitness=",pop[i].f," and is ", pop[i].d    
        
    return timeline
        
#---------------------------------------------------------------------------------------#
def Levy_Flight(func,pop,timeline,S,stats):  
    """
    Generate new children using Levy flights
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    pop : list of parent objects
        The current parents representing system designs
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
    pop : list of parent objects
        The proposed parent representing new system designs
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    # Initialize Variables
    feval=0
    used=[]          
    for i in range(0,int(S.fl*S.p),1):
        r=int(np.random.rand()*S.p)
        while r in used:
            r=int(np.random.rand()*S.p)
        used.append(r)
        pop.append(util.Parent(pop[r].f,cp.deepcopy(pop[r].d),pop[r].i))
        while pop[i+S.p].f>=pop[r].f: 
            dist=np.zeros(len(pop[r].d))   #Reset distance vector       
            # Calculate distance to each of the cities
            for j in range(0,len(pop[r].d),1):
                dist[j]=round(m.sqrt((pop[i+S.p].d[j][0]-pop[i+S.p].d[pop[i+S.p].i][0])**2 + \
                                     (pop[i+S.p].d[j][1]-pop[i+S.p].d[pop[i+S.p].i][1])**2))
                if j==pop[i+S.p].i:
                    dist[j]=1E99
                    
            #Based on Levy flights, determine next city to visit
            p=abs(sm.Levy(1))
            if p<1:
                ind=np.argmin(dist)  #Determine the index of the closest city
            elif p>=1:
                d=np.min(dist)*p   #Use Levy flight to determine maximum distance searched
                ind=np.where(dist==np.max(dist[dist <= d]))[0][0] #Determine the index of the furthest city within d
            else:
                print "An invalid value of p was generated using levy flights."
             
            # If the closest city and randomly chosen city are non-adjacent, invert the cities btwn closest (inclused) and 
            # current city (not included)
            if ind>pop[i+S.p].i+1 or ind<pop[i+S.p].i-1:
                if ind<pop[i+S.p].i:
                    pop[i+S.p].d[ind+1:pop[i+S.p].i+1]=reversed(pop[i+S.p].d[ind+1:pop[i+S.p].i+1])
                    (pop[i+S.p].f,g)=func(pop[i+S.p].d)
                else: 
                    pop[i+S.p].d[pop[i+S.p].i:ind]=reversed(pop[i+S.p].d[pop[i+S.p].i:ind])
                    (pop[i+S.p].f,g)=func(pop[i+S.p].d)
                feval+=1    #Update function eval counter
                stats.lfe+=1
                pop[i+S.p].i=ind   # For the ith parent, update the index of the most recently visited city
            else:
                pop[i+S.p].i=ind   # For the ith parent, update the index of the most recently visited city
                break#
        if pop[i+S.p].f<pop[r].f:#
            stats.lfc+=1#
            pop[r]=cp.deepcopy(pop[i+S.p])
                
    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval  
    del pop[S.p:len(pop)]    
    
    if S.d:
        print "\nAt end of Levy_Flight, the new proposed parents are: "
        for i in range(0,len(pop),1): 
            print "New parent #", i,":",pop[i].f,pop[i].d
      
    return pop,timeline,stats

#---------------------------------------------------------------------------------------#
def Crossover(func,pop,timeline,S,stats):  
    """
    Generate new parents using crossover strategies.  
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    pop : list of parent objects
        The current parent representing system designs
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
    pop : list of parent objects
        The proposed parent representing new system designs
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """ 
        
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    feval=0
    for i in range(0,int(S.fe*len(pop)),1):
        rand1=i
        
        #Randomly choose starting parent #2  
        rand2=int(np.random.rand()*len(pop))     
        while rand2==rand1:
            rand2=int(np.random.rand()*len(pop))

        # Loop over every city in the parent
        for c1 in range(0,len(pop[rand1].d),1):
            d2=(contains_sublist(pop[rand2].d,pop[rand1].d[c1])+1)%len(pop[rand1].d)
            d1=(contains_sublist(pop[rand1].d,pop[rand2].d[d2]))
            c2=(contains_sublist(pop[rand2].d,pop[rand1].d[(d1+1)%len(pop[rand1].d)]))%len(pop[rand1].d)
            # Test permutation in first parent
            tmp=cp.copy(pop[rand1].d)
            if c1<d1:
                tmp[c1+1:d1+1]=reversed(tmp[c1+1:d1+1])
            else:
                tmp[d1:c1]=reversed(tmp[d1:c1])

            # If improved, save; else test permutation in second parent
            (ftmp,g)=func(tmp)
            feval+=1
            stats.ce+=1
            if ftmp<pop[rand1].f:
                pop[rand1].d=cp.copy(tmp)
                pop[rand1].f=ftmp
                stats.cc+=1
            else:    
                tmp=cp.copy(pop[rand2].d)
                if c2<d2:
                    tmp[c2:d2]=reversed(tmp[c2:d2])
                else:
                    tmp[d2+1:c2+1]=reversed(tmp[d2+1:c2+1])
                (ftmp,g)=func(tmp)
                feval+=1
                stats.ce+=1
                if ftmp<pop[rand2].f:
                    pop[rand2].d=cp.copy(tmp) 
                    pop[rand2].f=ftmp
                    stats.cc+=1        
        
    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval 
    
    # Debug Statements
    if S.d:
        print "\nAt end of Study_Operator, the new proposed parents are: "
        for i in range(0,len(pop),1): 
            print "New parent #", i,":",pop[i].f,pop[i].d
            
    return pop,timeline,stats
    
#---------------------------------------------------------------------------------------#
def Two_opt(func,pop,timeline,S,stats):  
    """
    Generate new pop using Levy flights and two_opt operator.
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    pop : a list of parent objects
        The current parents representing system designs
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
    pop : list of parent objects
        The proposed parent representing new system design
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """ 
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    # Initialize variables
    feval=0
    for i in range(0,int(S.fe*len(pop)),1):    
        for a in range(0,len(pop[i].d),1):
            b=(a+1)%len(pop[i].d)  #City #2
            c=(a+2)%len(pop[i].d)    #City #3

            # Calculate distance to each of the cities
            dist=np.zeros(len(pop[i].d))   #Reset distance vector 
            for j in range(0,len(pop[i].d),1):
                dist[j]=round(m.sqrt((pop[i].d[j][0]-pop[i].d[b][0])**2 + (pop[i].d[j][1]-pop[i].d[b][1])**2))
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

            e=(d+1)%len(pop[i].d)    #City #5

            # Construct partial tour and determine length; test for coincidences to construct segments
            # Handle the special case
            if e==a:
                old_dist=round(m.sqrt((pop[i].d[b][0]-pop[i].d[c][0])**2 + (pop[i].d[b][1]-pop[i].d[c][1])**2) + \
                              m.sqrt((pop[i].d[a][0]-pop[i].d[d][0])**2 + (pop[i].d[a][1]-pop[i].d[d][1])**2) )
                new_dist=round(m.sqrt((pop[i].d[a][0]-pop[i].d[c][0])**2 + (pop[i].d[a][1]-pop[i].d[c][1])**2) + \
                              m.sqrt((pop[i].d[b][0]-pop[i].d[d][0])**2 + (pop[i].d[b][1]-pop[i].d[d][1])**2) )
                feval+=1  
                stats.aoe+=1

                # Compare path lengths, change parent if new path is shorter
                if new_dist<old_dist:
                    hold=pop[i].d[b]
                    del pop[i].d[b]
                    pop[i].d.insert(a,hold)    

                    # Update the function eval counter in the timeline
                    (pop[i].f,g)=func(pop[i].d)
                    feval+=1  
                    stats.aoe+=1
                    stats.aoc+=1

            # Ignore all of the cases where the distance is the same       
            elif d!=a and d!=c and e!=b:
                old_dist=round(m.sqrt((pop[i].d[a][0]-pop[i].d[b][0])**2 + (pop[i].d[a][1]-pop[i].d[b][1])**2) + \
                              m.sqrt((pop[i].d[b][0]-pop[i].d[c][0])**2 + (pop[i].d[b][1]-pop[i].d[c][1])**2) + \
                              m.sqrt((pop[i].d[d][0]-pop[i].d[e][0])**2 + (pop[i].d[d][1]-pop[i].d[e][1])**2) )
                new_dist=round(m.sqrt((pop[i].d[a][0]-pop[i].d[c][0])**2 + (pop[i].d[a][1]-pop[i].d[c][1])**2) + \
                              m.sqrt((pop[i].d[d][0]-pop[i].d[b][0])**2 + (pop[i].d[d][1]-pop[i].d[b][1])**2) + \
                              m.sqrt((pop[i].d[b][0]-pop[i].d[e][0])**2 + (pop[i].d[b][1]-pop[i].d[e][1])**2) )
                feval+=1  
                stats.aoe+=1

                # Compare path lengths, change parent if new path is shorter
                if new_dist<old_dist:
                    hold=pop[i].d[b]
                    del pop[i].d[b]
                    if b<e:
                        pop[i].d.insert(d,hold)
                    else:
                        pop[i].d.insert(e,hold)

                    # Update the function eval counter in the timeline
                    (pop[i].f,g)=func(pop[i].d)
                    feval+=1
                    stats.aoe+=1
                    stats.aoc+=1
    
    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval 
    
    # Debug Statements
    if S.d:
        print "\nAt end of Two_opt, the new proposed parents are: "
        print "New parent :",pop[i].f,pop[i].d
          
    return pop,timeline,stats

#---------------------------------------------------------------------------------------#
def Three_Opt(func,pop,timeline,S,stats):  
    """
    Generate new parents using Levy flights
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    pop : a parents object
        The current parents representing system designs
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
    pop : parent object
        The proposed parent representing new system design
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    # Initialize variables
    feval=0
    for i in range(0,len(pop),1): 
        tmp=[]   # Make a local copy of current parent designs

        # Generate 3 random nodes
        breaks=np.sort(np.random.rand(3)*len(pop[i].d)//1+1)%len(pop[i].d)
        #Ensure that 3 different nodes are selected
        while breaks[1]==breaks[0] or breaks[1]==breaks[2]: 
            breaks[1]=(np.random.rand()*len(pop[i].d)//1+1)%len(pop[i].d) 
            breaks=np.sort(breaks)
        while breaks[2]==breaks[0]:
            breaks[2]=(np.random.rand()*len(pop[i].d)//1+1)%len(pop[i].d) 
        breaks=np.sort(breaks)   

        # Make reconnections first way
        tmp[0:int(breaks[0])]=pop[i].d[0:int(breaks[0])]
        tmp[len(tmp):int(len(tmp)+breaks[2]-breaks[1])]=pop[i].d[int(breaks[1]):int(breaks[2])]
        tmp[len(tmp):int(len(tmp)+breaks[1]-breaks[0])]=pop[i].d[int(breaks[0]):int(breaks[1])]
        tmp[len(tmp):int(len(tmp)+breaks[2]-len(pop[i].d))]=pop[i].d[int(breaks[2]):len(pop[i].d)]

        # Test new route against old; update if better
        (ftmp,g)=func(tmp)
        feval+=1
        stats.toe+=1
        if ftmp<pop[i].f:
            pop[i].d=cp.copy(tmp) 
            pop[i].f=ftmp
            stats.toc+=1

        # Make reconnections second way
        tmp=[]   # Reset local copy of current parent designs
        tmp[0:int(breaks[0])]=pop[i].d[0:int(breaks[0])]
        tmp[len(tmp):int(len(tmp)+breaks[1]-breaks[0])]=reversed(pop[i].d[int(breaks[0]):int(breaks[1])])
        tmp[len(tmp):int(len(tmp)+breaks[2]-breaks[1])]=reversed(pop[i].d[int(breaks[1]):int(breaks[2])])
        tmp[len(tmp):int(len(tmp)+breaks[2]-len(pop[i].d))]=pop[i].d[int(breaks[2]):len(pop[i].d)]

        # Test new route against old; update if better
        (ftmp,g)=func(tmp)
        feval+=1
        stats.toe+=1
        if ftmp<pop[i].f:
            pop[i].d=cp.copy(tmp) 
            pop[i].f=ftmp
            stats.toc+=1
        
    # Update the function eval counter in the timeline
    (pop[i].f,g)=func(pop[i].d)
    feval+=1
    timeline[-1].e=timeline[-1].e+feval 
    
    # Debug Statements
    if S.d:
        print "\nAt end of 3-opt, the new proposed parent are: "
        print "New parent :",pop[i].f,pop[i].d
    
    return pop,timeline,stats

#---------------------------------------------------------------------------------------#               
def Flight_Operation(pop,n,prob_in,debug=False):
    """
    Generate new parents
   
    Parameters
    ==========
    pop : list of parent objects
        The current parent representing system designs
    n : integer    
        Number of parents to generate
    prob_in : Object
        An object containing the problem information
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)  
   
    Returns
    =======
    pop : list of parent objects
        The new parent representing system designs
    """
 
    # Initialize parents with random initial solution based on Levy flights
    for i in range(0,n,1):
        tmp=cp.copy(prob_in.n)   #Local copy of the cities
        rand=int(np.random.rand()*len(tmp))  #Randomly choose starting parent
        pop.append(util.Parent(5E20,[]))  #Create an instance of parent object
        pop[i].d.append(list(tmp[rand])) #Final route -> append starting parent
        tmp=np.delete(tmp,rand,0)   #Delete the chosen parent from the temporary copy so it is not visited again
        
        # Loop over the list of cities until all have been visited
        while len(tmp) != 0:
            dist=np.zeros(len(tmp))   #Reset distance vector
            # Calculate distance to each of the remaining cities
            for j in range(0,len(tmp),1):
                dist[j]=round(m.sqrt((tmp[j][0]-pop[i].d[-1][0])**2 + (tmp[j][1]-pop[i].d[-1][1])**2))
            #Based on Levy flights, determine next city to visit
            p=abs(sm.Levy(1))
            if p<1:
                pop[i].d.append(list(tmp[np.argmin(dist)]))
                tmp=np.delete(tmp,np.argmin(dist),0)
            elif p>=1:
                d=np.min(dist)*p
                pop[i].d.append(tmp[np.where(dist==np.max(dist[dist <= d]))[0]][0].reshape(2).tolist())
                tmp=np.delete(tmp,np.where(dist==np.max(dist[dist <= d]))[0][0],0)
            else:
                print "An invalid value of p was generated using levy flights."
    
    if debug:
        print 'After flight operations:' 
        for i in range(n) :    
            print "Parent ",i,": Fitness=",pop[i].f,", Design=",pop[i].d
            
    return pop   

#---------------------------------------------------------------------------------------#               
def Partial_Inversion(func,pop,n,timeline,stats,debug=False):
    """
    Improve population using partial inversion
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    pop : list of parent objects
        The current parents representing system designs
    n : integer    
        Number of parents to generate
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
    pop : list of parent objects
        The new parent representing system designs
    timeline : list of history objects
        Updated histories of the optimization process 
    stats : Object    
        Updated object that tracks the function eval and replacement statistics to evaulate effectiveness
    """  
    
    feval=0
    for i in range(0,n,1):
        tmp=cp.deepcopy(pop[i])   #Local copy 
        
        # Randomly establish starting point
        rand=int(np.random.rand()*len(tmp.d))  
        order=[]
        order[0:(len(tmp.d)-rand)]=range(rand,len(tmp.d),1)
        order[(len(tmp.d)-rand):len(tmp.d)]=range(0,rand,1)
        pop[i].i=order[-1]
        
        # Visit each city in the order specified
        while len(order)!=0: 
            dist=np.zeros(len(tmp.d))   #Reset distance vector
            tmp=cp.deepcopy(pop[i])   #Reset the local copy of the parent object 
            
            # Calculate distance to each of the cities
            for j in range(0,len(tmp.d),1):
                dist[j]=round(m.sqrt((tmp.d[j][0]-tmp.d[order[0]][0])**2 + (tmp.d[j][1]-tmp.d[order[0]][1])**2))
                if j==order[0]:
                    dist[j]=1E99
            
            # If the closest city and randomly chosen city are non-adjacent, invert the cities btwn closest (included) and 
            #furthest city (not included)
            imin=np.argmin(dist)  #Determine the index of the closest city
            pop[i].i=imin   # For the ith parent, update the index of the most recently visited city
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
                if tmp.f<pop[i].f:
                    pop[i]=cp.deepcopy(tmp)
                    stats.pic+=1
            # Advance to the next city
            del order[0]
            
    # Update the function eval counter in the timeline
    timeline[-1].e=timeline[-1].e+feval 
    
    if debug:
        print 'After partial inversion:' 
        for i in range(n) :    
            print "Parent ",i,": Fitness=",pop[i].f,", Design=",pop[i].d  
            
    return pop,timeline,stats
    
#---------------------------------------------------------------------------------------#               
def Read_TSP(filename,debug=False):
    """
    Read the starting TSP points.
   
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
    levy_flight_changes : integer
        The number of time the levy_flight method updated the current solution
    levy_flight_fevals : integer
        The number of time the levy_flight method performed a function eval
    crossover_changes : integer
        The number of time the crossover method updated the current solution
    crossover_fevals : integer
        The number of time the crossover method performed a function eval
    a_operator_changes : integer
        The number of time the a_operator method updated the current solution
    a_operator_fevals : integer
        The number of time the a_operator method performed a function eval
    three_opt_changes : integer
        The number of time the three_opt method updated the current solution
    three_opt_fevals : integer
        The number of time the three_opt method performed a function eval
    Returns
    =======
    None
    """
        
    def __init__(self,partial_inversion_changes=0,partial_inversion_fevals=0,levy_flight_changes=0,levy_flight_fevals=0, \
                 crossover_changes=0,crossover_fevals=0,a_operator_changes=0,a_operator_fevals=0,\
                 three_opt_changes=0,three_opt_fevals=0):
        self.pic=partial_inversion_changes
        self.pie=partial_inversion_fevals
        self.lfc=levy_flight_changes
        self.lfe=levy_flight_fevals
        self.cc=crossover_changes
        self.ce=crossover_fevals
        self.aoc=a_operator_changes
        self.aoe=a_operator_fevals
        self.toc=three_opt_changes
        self.toe=three_opt_fevals