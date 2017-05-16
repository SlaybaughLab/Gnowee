################################################################################
#
# Module : DiscreteCuckooSearch.py
#
# Contains : Functions and methods for a Discrete Cuckoo Search Algorithm developed by Ouaarab.
#            to perform combinatorial (TSP) optimization problems.  The original  
#            algorithm is given in "Discrete Cuckoo Search Algorithm for the Travelling Salesman Problem".             
#
# Author : James Bevins
#
# Last Modified: 14Oct15
#
####################################################################################

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
def DCS(func,S,tsp):
    """
    Perform a discrete cuckoo search optimization (CCS) based on algorithm in 
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
    
    # Initialize nests with random initial solution; then determine shortest route based on city pair distances
    for i in range(0,S.c,1):
        tmp=cp.copy(tsp.n)
        nests.append(csl.Nest(5E20,[]))
        rand=int(len(tmp)*np.random.rand())
        nests[i].d.append(tmp[rand])
        del tmp[rand]
        while len(tmp) > 0: 
            best=(0,1E10)
            for j in range(0,len(tmp),1):
                dx=m.sqrt((tmp[j][0]-nests[i].d[-1][0])**2 + (tmp[j][1]-nests[i].d[-1][1])**2)
                if dx < best[1]:
                    best=(j,dx)
            nests[i].d.append(tmp[best[0]])
            del tmp[best[0]] 
    # Initialize nests with random initial solutions
#    for i in range(0,S.c,1):
#        tmp=cp.copy(tsp.n)
#        nests.append(csl.Nest(5E20,[]))
#        while len(tmp) > 0:
#            rand=int(len(tmp)*np.random.rand())
#            nests[i].d.append(tmp[rand])
#            del tmp[rand]
    if S.d:
        print 'Initial nests:' 
        for i in range(S.c) :    
            print "Nest ",i,": Fitness=",nests[i].f,", Design=",nests[i].d 
    
    # Calculate initial fitness values
    (nests,timeline)=csl.Get_Best_Nest(obj,nests,nests,timeline,S)
    
    if True:
        print 'Initial nests:' 
        for i in range(S.c) :    
            print "Nest ",i,": Fitness=",nests[i].f,", Design=",nests[i].d 
    
    # Iterate until termination criterion met
    converge = False
    while timeline[-1].g <= S.gm and timeline[-1].e <= S.em and converge==False:
        
        # Local search using Levy flights and evaluate fitness
        new_nests=Top_Nests(nests,S)
        (nests,timeline)=csl.Get_Best_Nest(obj,nests,new_nests,timeline,S,random_replace=True)
        
        # Global search using discovery
        (nests,timeline)=Discard_Nests(obj,nests,S,timeline)        
        
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
        if ((timeline[-1].f-S.of)/S.of) <= S.ot:
            converge=True
            print "Fitness Convergence"  
#!!        print timeline[-1].g
    #Determine execution time
    print "Program execution time was %f."  % (time.time() - start_time)

    print "\nAt end of optimization:"      
    for i in range(len(nests)):
        print "The new nests #",i,"has fitness=",nests[i].f," and is ", nests[i].d    
        
    return timeline
        
#---------------------------------------------------------------------------------------#
def Top_Nests(nests,S,max_moves=4):  
    """
    Generate new nests using Levy flights
   
    Parameters
    ==========
    nests : list of nest objects
        The current nests representing system designs
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
    max_moves : integer
        Maximum number of 2-opt moves in interval (Default=4)
   
    Returns
    =======
    tmp : list of nest objects
        The proposed nests representing new system designs
    """
    
    assert 0<=S.pt<=1, 'pt must be between 0 and 1 inclusive in Get_Nests function.'
    
    # Initialize variable
    tmp=[]  # Local copy of nest that is modified
    sub_int=1./(float(max_moves)+1)   # Calculate the sub intervals for the TLF (0,1) sample interval
    k=np.random.rand(S.c)<S.pt # Detemine which nests to select
    j=0

    # Always search from best position (sometimes will result in > pt nests selected)
    k[0]=True
    
    for i in range(0,len(k),1):
        if k[i]==True:
            step=sm.TLF()[0]  # Sample TLF to determine number of steps      
            tmp.append(cp.copy(nests[i]))   # Make a local copy of current nest 
            if S.d:
                print "\nIn Top_Nests, the selected nests are: "
                print "Selected nest #", i,":",tmp[j].d
                print "The step is", step
            if step < 0.8:
                tmp[j].d=Two_Opt(tmp[j],int(step/sub_int)+1)
            elif step < 1.0:
                tmp[j].d=Double_Bridge(tmp[j])
            else:
                "Step is greater than 1 (%f).  Call an exterminator to fix this bug!" %step
            j+=1
            
    if S.d:
        print "\nAt end of Top_Nests, the new proposed nests are: "
        for i in range(0,len(tmp),1): 
            print "New nest #", i,":",tmp[i].d
      
    return tmp
            
#---------------------------------------------------------------------------------------#               
def Discard_Nests(func,nests,S,timeline):
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
    assert S.pa>0 and S.pa <=1, 'The probability that a nest is discovered must exist on (0,1]'

    # Initialize variable  
    feval=0
    max_moves=4#!!
    sub_int=1./(float(max_moves)+1)   # Calculate the sub intervals for the TLF (0,1) sample interval  
    
    # First remove any duplicates up to S.pa * S.c
    seen = set()
    count=0
    for i in range(0,S.c,1):
        if nests[i].f in seen:
            step=sm.TLF()[0]  # Sample TLF to determine number of steps      
            if step < 0.8:
                nests[i].d=Two_Opt(nests[i],int(step/sub_int)+1)
            elif step < 1.0:
                nests[i].d=Double_Bridge(nests[i])
            else:
                "Step is greater than 1 (%f).  Call an exterminator to fix this bug!" %step
            (fnew,gnew)=func(nests[i].d)
            feval += 1
            nests[i].f=fnew
            count+=1
        elif nests[i].f not in seen:
            seen.add(nests[i].f)
        if count >= int(S.pa*S.c):
            break

#Try#3    
#    k=np.random.rand(S.c)<S.pa # Detemine which nests to select
# Always search from best position (sometimes will result in < pa nests selected)
#    k[0]=False
#    for i in range(0,S.c,1):
#        if k[i]==True:
#            step=sm.TLF()[0]  # Sample TLF to determine number of steps      
#            if step < 0.8:
#                nests[i].d=Two_Opt(nests[i],int(step/sub_int)+1)
#            elif step < 1.0:
#                nests[i].d=Double_Bridge(nests[i])
#            else:
#                "Step is greater than 1 (%f).  Call an exterminator to fix this bug!" %step
#            (fnew,gnew)=func(nests[i].d)
#            feval += 1
#            nests[i].f=fnew            

#Try#2        
    for i in range(count,int(S.pa*S.c),1):
#        j=int(S.pt*S.c+S.pa*S.c*np.random.rand())
        j=int(1+(S.c-1)*np.random.rand())
        tmp=cp.copy(nests[j].d)
        nests[j].d=[]
        rand=int(len(tmp)*np.random.rand())
        nests[j].d.append(tmp[rand])
        del tmp[rand]
        while len(tmp) > 0: 
            best=(0,1E10)
            for k in range(0,len(tmp),1):
                dx=m.sqrt((tmp[k][0]-nests[j].d[-1][0])**2 + (tmp[k][1]-nests[j].d[-1][1])**2)
                if dx < best[1]:
                    best=(k,dx)
            nests[j].d.append(tmp[best[0]])
            del tmp[best[0]] 

#Try#1
#    for j in range(len(nests)-1,int(len(nests)*S.pt),-1):  #!!! Not right!
#        tmp=cp.copy(nests[j].d)
#        nests[j].d=[]
#        while len(tmp) > 0:
#            rand=int(len(tmp)*np.random.rand())
#            nests[j].d.append(tmp[rand])
#            del tmp[rand] 
            
        (fnew,gnew)=func(nests[j].d)
        feval += 1
        nests[j].f=fnew

    #Sort the nests
    nests.sort(key=lambda x: x.f)
        
    #Store history on timeline if new optimal design found
    if nests[0].f<timeline[-1].f and (timeline[-1].f-nests[0].f)/nests[0].f > S.ct:
        timeline.append(csl.Event(timeline[-1].g,timeline[-1].e+feval,nests[0].f,nests[0].d))
    else:
        timeline[-1].e+=feval
        
    if S.d:
        print "\nAt end of function Empty_Nests:"      
        for i in range(len(nests)):
            print "The new nests #",i,"has fitness=",nests[i].f," and is ", nests[i].d
    return nests,timeline     

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
                    node,x,y = line.split(" ")         
                    prob.n.append([int(x.strip()),int(y.strip())]) 
    
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
def Double_Bridge(nests,num_steps=1):
    """
    Perform a double bridge change to the input problem
    
    Parameters
    ==========
    nests : Nest Object
        A Nest object containing the problem information
   
    Optional
    ========
    num_steps : Integer
        Number of double bridge steps to perform
   
    Returns
    =======
    tmp : Array
        An array containing the problem design solution that has been updated by num_steps 
        double bridge moves
    """
    
    tmp=[]   # Make a local copy of current nest designs
    
    # Perform double bridge move, num_step times
    for i in range(0,num_steps,1):
        #Generate 4 random nodes
        breaks=np.sort((np.random.rand(4)*len(nests.d)//1+1)%len(nests.d))
        
        #Ensure that 4 different nodes are selected
        while breaks[1]==breaks[0] or breaks[1]==breaks[2] or breaks[1]==breaks[3]:
            breaks[1]=np.sort((np.random.rand()*len(nests.d)//1+1)%len(nests.d))    
        while breaks[2]==breaks[0] or breaks[2]==breaks[3]:
            breaks[2]=np.sort((np.random.rand()*len(nests.d)//1+1)%len(nests.d)) 
        while breaks[3]==breaks[0]:
            breaks[3]=np.sort((np.random.rand()*len(nests.d)//1+1)%len(nests.d)) 
        np.sort(breaks)     
        
        # Make reconnections
        tmp.append(nests.d[0:int(breaks[0])])
        tmp.append(nests.d[int(breaks[2]):int(breaks[3])])
        tmp.append(nests.d[int(breaks[1]):int(breaks[2])])
        tmp.append(nests.d[int(breaks[0]):int(breaks[1])])
        tmp.append(nests.d[int(breaks[3]):len(nests.d)])

        #Make first set of reconnections
#        if break1!=(len(tmp)-1) and break3!=(len(tmp)-1):
#            hold=tmp[break1+1]
#            tmp[break1+1]=tmp[break3+1]
#            tmp[break3+1]=hold
#        elif break3!=(len(tmp)-1) :
#            hold=tmp[0]
#            tmp[0]=tmp[break3+1]
#            tmp[break3+1]=hold 
#        elif break1!=(len(tmp)-1):
#            hold=tmp[break1+1]
#            tmp[break1+1]=tmp[0]
#           tmp[0]=hold
#        else:
#            print "Impossible!  Break1 != Break3 != the last element at the same time."
            
        #Make second set of reconnections
#        if break2!=(len(tmp)-1) and break4!=(len(tmp)-1):
#            hold=tmp[break2+1]
#            tmp[break2+1]=tmp[break4+1]
#            tmp[break4+1]=hold
#        elif break2!=(len(tmp)-1):
#            hold=tmp[break2+1]
#            tmp[break2+1]=tmp[0]
#            tmp[0]=hold
#        elif break4!=(len(tmp)-1):
#            hold=tmp[0]
#            tmp[0]=tmp[break4+1]
#            tmp[break4+1]=hold
#        else:
#            print "Impossible!  Break1 != Break3 != the last element at the same time."
            
    return sum(tmp,[])  

#---------------------------------------------------------------------------------------#
def Two_Opt(nests,num_steps=1):
    """
    Perform 2-opt changes to the input problem
    
    Parameters
    ==========
    nests : Nest Object
        A Nest object containing the problem information
   
    Optional
    ========
    num_steps : Integer
        Number of two opt steps to perform
   
    Returns
    =======
    tmp : Array
        An array containing the problem design solution that has been updated by num_steps 
        2-opt moves
    """

    tmp=cp.copy(nests.d)   # Make a local copy of current nest designs
 
    # Perform 2-opt move, num_step times
    for i in range(0,num_steps,1):
        breaks=np.sort((np.random.rand(2)*len(tmp)//1+1)%len(tmp))
        while breaks[0]==breaks[1]:
            breaks[1]=np.sort((np.random.rand()*len(tmp)//1+1)%len(tmp))
            np.sort(breaks)         
#        if breaks[1]==(len(tmp)-1):
#            a[0:int(breaks[0]+1)]=reversed(a[int(breaks[0]+1):int(breaks[1]+1)])
#            hold=tmp[0]
#            tmp[0]=tmp[break2]
#            tmp[break2]=hold
#        else:
#            hold=tmp[break1+1]
#            tmp[break1+1]=tmp[break2]
#            tmp[break2]=hold
        tmp[int(breaks[0]):int(breaks[1])]=reversed(tmp[int(breaks[0]):int(breaks[1])])
    
    return tmp