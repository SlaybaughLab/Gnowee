#######################################################################################################
#
# Module : Gnowee.py
#
# General nearly-global metaheuristic optimization algorithm. Uses a blend of common heuristics to solve 
# difficult gradient free constrained MINLP problems with categorical variables. It is capable of solving 
# simpler problems, but may not be the algorithm of choice. 
#
# Author : James Bevins
#
# Last Modified: 18Apr17
#
#######################################################################################################

from GnoweeHeuristics import Disc_Levy_Flight,Cont_Levy_Flight, Crossover, Cont_Crossover, Mutate, ScatterSearch 
from GnoweeHeuristics import Initialize, ScatterSearch2
import SamplingMethods as sm
import GnoweeUtilities as util
import numpy as np
import math as m
import time

#---------------------------------------------------------------------------------------#
def main(func,lb,ub,varType,S,discreteVals=[]):
    """
    Main program for the optimization. 
    
    Parameters
    ==========
    func : function
        The objective function to be minimized
    lb : list or array
        The lower bounds of the design variable(s). Only enter the bounds for continuous and integer/binary variables.
    ub : list or array
        The upper bounds of the design variable(s). Only enter the bounds for continuous and integer/binary variables.
    varType : list or array
        The type of variable for each position in the upper and lower bounds array. Discrete variables are to be included 
        last as they are specified separatly from the lb/ub throught the discreteVals optional input. A variable can have two 
        types (for example, 'dx' could denote a layer that can take multiple materials and be placed at multiple design locations)
        Allowed values:
        'c' = continuous
        'i' = integer/binary (difference denoted by ub/lb)
        'd' = discrete where the allowed values are given by the option discreteVals nxm arrary with n=# of discrete variables
              and m=# of values that can be taken for each variable 
        'x' = combinatorial.  All of the variables denoted by x are assumed to be "swappable" in combinatorial permutations.  
              There must be at least two variables denoted as combinatorial.  
        'f' = fixed design variable 
    S : Object    
        An object representing the settings for the Gnowee optimization algorithm
   
    Optional
    ========
    discreteVals : list of list(s)
        nxm with n=# of discrete variables and m=# of values that can be taken for each variable 
        Default=[[]]
   
    Returns
    =======
    timeline : list
        Storage list for design event objects for the current top solution vs generation. Only stores the information when 
        new optimal designs are found.
    """
    
    start_time=time.time()     #Start Clock
    timeline=[]                #List of history objects
    pop=[]                     #List of parent objects
    
    # Check input variables and convert to numpy arrays where required
    assert varType.count('d')==len(discreteVals), 'The allowed discrete values must be specified for each discrete variable. \
                                                  {} in varType, but {} in discreteVals.'.format(varType.count('d'),len(discreteVals))
    for d in range(len(discreteVals)):
        lb.append(0)
        ub.append(len(discreteVals[d])-1)
        discreteVals[d]=np.array(discreteVals[d])
    lb = np.array(lb)
    ub = np.array(ub)
    assert set(varType).issubset(['c','i','d','x','f']), 'The variable specifications do not match the allowed values of \
                                                               "c", "i", or "d".  The varTypes specified is {}'.format(varType)
    assert len(lb)==len(ub), 'Lower and upper-bounds must be the same length'
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
    assert len(lb)==len(varType), 'Valid number of values must be specified for each design variable type. Currently, the size of \
                                  the bounds are {}, and the size of the variable types is {}.'.format(len(lb),len(varType))
        
    # Check for objective function(s) 
    assert hasattr(func, '__call__'), 'Invalid function handle'
    obj = lambda x,penalty: func(x,penalty)  
    
    # Initialize population with random initial solutions
    initNum = max(S.p*2,len(ub)*10)
    initParams = Initialize(initNum, S.s, lb, ub, varType)
    for p in range(0,initNum,1):
        pop.append(util.Parent(1E99,initParams[p]))  
    # Develop ID vectors for each variable type
    cID=[]
    iID=[]
    dID=[]
    xID=[]
    for var in range(len(varType)):
        if 'c' in varType[var]:
            cID.append(1)
        else:
            cID.append(0)
        if 'i' in varType[var]:
            iID.append(1)
        else:
            iID.append(0)
        if 'd' in varType[var]:
            dID.append(1)
        else:
            dID.append(0)
        if 'x' in varType[var]:
            xID.append(1)
        else:
            xID.append(0)
    cID=np.array(cID)
    iID=np.array(iID)
    dID=np.array(dID)
    xID=np.array(xID)
            
    # Calculate initial fitness values and trim population to S.p members
    (pop,changes,timeline)=util.Get_Best(obj,pop,[p.d for p in pop],lb,ub,varType,timeline,S,1)
    pop=pop[0:S.p]    
#    dim=3
#    perm=(np.random.permutation(range(S.p-dim))+dim)
#    ind=[i for i in range(dim)]+[perm[i] for i in range(dim)]
#    tmp=[pop[i] for i in ind]    
#    import copy as cp
#    pop=cp.deepcopy(tmp)
    
    nDLF=0
    cDLF=0
    nCLF=0
    cCLF=0
    nCC=0
    cCC=0
    nSS=0
    cSS=0
    nC=0
    cC=0
    nM=0
    cM=0
    
    # Iterate until termination criterion met
    converge = False
    while timeline[-1].g <= S.gm and timeline[-1].e <= S.em and converge==False:
        S.fd=np.random.rand()/2.5
        S.fe=np.random.rand()/2.5
        S.fl=np.random.rand()/2.5
        # Levy flights
        if sum(iID)+sum(dID)>=1 and sum(cID)>=1:
            #print "pre Disc_Levy_Flight:", [p.d[3:] for p in pop]
            (d_children,dind)=Disc_Levy_Flight([p.d for p in pop],lb,ub,iID+dID,S)
            #print "post Disc_Levy_Flight:", [c[3:] for c in d_children], '\n'     
#            (pop,changes,timeline)=util.Get_Best(obj,pop,children,lb,ub,varType, timeline,S,0,indices=ind,mhFrac=S.fd,random_replace=False, discreteID=dID, discreteMap=discreteVals)
#            nDLF+=len(children)
#            cDLF+=changes
            #print "pre Cont_Levy_Flight:", [p.d[3:] for p in pop]
            (c_children,cind)=Cont_Levy_Flight([p.d for p in pop],lb,ub,cID,S)
            #print "post Cont_Levy_Flight:", [c[3:] for c in c_children], '\n'     
            children=[]
            ind=[]
            for i in range(0,len(cind)):
                if cind[i] in dind:
                    t=dind.index(cind[i])
                    children.append(d_children[t]*(iID+dID)+c_children[i]*cID)
                    ind.append(cind[i])
                    del d_children[t]
                    del dind[t]
                else:
                    children.append(c_children[i])
                    ind.append(cind[i])
            for i in range(len(dind)):
                children.append(d_children[i])
                ind.append(dind[i])      
            (pop,changes,timeline)=util.Get_Best(obj,pop,children,lb,ub,varType, timeline,S,0,indices=ind,mhFrac=0.2,random_replace=False, discreteID=dID, discreteMap=discreteVals)   
        
        elif sum(cID)>=1 and sum(iID)+sum(dID)==0:
            #print "pre Cont_Levy_Flight:", [p.d[3:] for p in pop]
            (children,ind)=Cont_Levy_Flight([p.d for p in pop],lb,ub,cID,S) 
            #print "post Cont_Levy_Flight:", [c[3:] for c in children], '\n'      
            (pop,changes,timeline)=util.Get_Best(obj,pop,children,lb,ub,varType, timeline,S,0,indices=ind,mhFrac=0.2,random_replace=True, discreteID=dID, discreteMap=discreteVals)   
            nCLF+=len(children)
            cCLF+=changes    
        
        # Crossover
        #print "pre Cont_Crossover:", [p.d[3:] for p in pop]
        (children,ind) = Cont_Crossover([p.d for p in pop],lb,ub,cID,S,intDiscID=iID+dID)
        #print "post Cont_Crossover:", [c[3:] for c in children], '\n'
        (pop,changes,timeline)=util.Get_Best(obj,pop,children,lb,ub,varType,timeline,S,0,discreteID=dID,discreteMap=discreteVals)
        nCC+=len(children)
        cCC+=changes
        
#        (children, changes, timeline) = ScatterSearch2(obj,pop,lb,ub,cID,timeline,S,discreteID=dID,discreteMap=discreteVals, intDiscID=iID+dID)
#        nSS+=6
#        cSS+=changes
        
        #print "pre ScatterSearch:", [p.d[3:] for p in pop]
        (children,ind) = ScatterSearch([p.d for p in pop],lb,ub,cID,S,intDiscID=iID+dID)
        #print "post ScatterSearch:", [c[3:] for c in children], '\n'
        (pop,changes,timeline)=util.Get_Best(obj,pop,children,lb,ub,varType, timeline,S,0,indices=ind,discreteID=dID,discreteMap=discreteVals)
        nSS+=len(children)
        cSS+=changes
        
        if sum(iID)+sum(dID)>=1 and sum(cID)>=1:
            #print "pre Crossover:", [p.d[3:] for p in pop]
            children = Crossover([p.d for p in pop],S)
            #print "post Crossover:", [c[3:] for c in children], '\n'
            (pop,changes,timeline)=util.Get_Best(obj,pop,children,lb,ub,varType,timeline,S,0,discreteID=dID,discreteMap=discreteVals)
            nC+=len(children)
            cC+=changes
                                     
        # Mutation
        if sum(cID)+sum(iID)+sum(dID)>=1:
            #print "pre Mutate:", [p.d[3:] for p in pop]
            children=Mutate([p.d for p in pop],lb,ub,cID,S,intDiscID=iID+dID)
            #print "post Mutate:", [c[3:] for c in children]
            (pop,changes,timeline)=util.Get_Best(obj,pop,children,lb,ub,varType,timeline,S,1,random_replace=False,discreteID=dID, discreteMap=discreteVals)  
            #print "post Mutate Update:", [p.d[3:] for p in pop], '\n'
            nM+=len(children)
            cM+=changes 
        
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

    # Print Stats
#    print "DLF: ", float(cDLF)/nDLF, nDLF
#    print "CLF: ", float(cCLF)/nCLF, nCLF
#    print "CC: ", float(cCC)/nCC, nCC
#    print "SS: ", float(cSS)/nSS, nSS
#    print "C: ", float(cC)/nC, nC
#    print "M: ", float(cM)/nM, nM
    
    d=np.array([pop[0].d])
    f=np.array(pop[0].f)
    for i in range(1,len(pop)):
        d=np.append(d,[pop[i].d],axis=0)
        f=np.append(f,pop[i].f)
#    print d
#    print f
#    print S.pen
    
    #Determine execution time
    print "Program execution time was %f."  % (time.time() - start_time)
    return timeline   

if __name__ == '__main__':
    main()