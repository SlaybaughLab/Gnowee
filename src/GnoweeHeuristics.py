#######################################################################################################
#
# Module : GnoweeHeuristics.py
#
# Heuristics supporting the Gnowee metaheuristic optimization algorithm. 
#
# Author : James Bevins
#
# Last Modified: 18Apr17
#
#######################################################################################################

import numpy as np
import copy as cp
import math as m

from scipy.stats import rankdata
from SamplingMethods import Levy, TLF, Initial_Samples
from itertools import combinations
from GnoweeUtilities import Rejection_Bounds, Simple_Bounds, Get_Best

#---------------------------------------------------------------------------------------#
def Initialize(numSamples, sampleMethod, lb, ub, varType):
    """
    Initialize a population.

    Parameters
    ==========
    numSamples : integer
        The number of samples to be generated
    sampleMethod : string
        The name of the sampling method to be used.  Options are 'random', 'nolh',
        'nolh-rp', 'nolh-cdr', or 'lhc'.
    pop : list of arrays
        The current parent sets of design variables representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    varType : array
        The type of variable for each design parameter.

    Returns
    =======
    initSamples : list of arrays
        The initialized set of samples
    """
    initSamples = Initial_Samples(lb, ub, sampleMethod, numSamples)
    for var in range(len(varType)):
        if varType[var]=='i' or varType[var]=='d':
            initSamples[:,var]=np.rint(initSamples[:,var])
    return initSamples
#---------------------------------------------------------------------------------------#
def Disc_Levy_Flight(pop,lb,ub,varID,S):  
    """
    Generate new children using Levy flights
   
    Parameters
    ==========
    pop : list of arrays
        The current parent sets of design variables representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    varID : array
        A truth array indicating the location of the variables to be permuted
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
   
    Returns
    =======
    children : list of arrays
        The proposed children sets of design variables representing new system designs
    used : list
        A list of the identities of the chosen index for each child
    """
    
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Levy_Flight function.'
    assert len(lb)==len(pop[0]), 'Bounds and pop have different #s of design variables in Levy_Flight function.' 
    assert len(lb)==len(varID), 'The bounds size ({}) must be consistent with the size of the variable ID truth vectorin Levy_Flight function.'.format(len(lb),len(varID))
    assert S.fl>=0 and S.fl <=1, 'The probability that a Levy flight is performed must exist on (0,1]'
    
    children=[] # Local copy of children generated
            
    # Determine step size using Levy Flight
    step=TLF(len(pop),len(pop[0]),alpha=S.a,gamma=S.g) 
    
    # Initialize Variables
    feval=0
    used=[] 
    for i in range(0,int(S.fl*S.p),1):
        k=int(np.random.rand()*S.p)
        while k in used:
            k=int(np.random.rand()*S.p)
        used.append(k)
        children.append(cp.deepcopy(pop[k])) 
        #print "pre:", children[i][3:]
        
        #Calculate Levy flight
        stepsize=np.round(step[k,:]*varID*(ub-lb))
        #print "stepsize:", stepsize[3:]
        if all(stepsize == 0):
            stepsize=np.round(np.random.rand(len(varID))*(ub-lb))*varID
            #print "mod stepsize:", stepsize[3:]
        children[i]=(children[i]+stepsize)%(ub+1-lb)
        #print "tmp:", children[i][3:]
        
        #Build child applying variable boundaries 
        children[i]=Rejection_Bounds(pop[k],children[i],stepsize,lb,ub,S)  
        #print "post:", children[i][3:], "\n"
      
    return children, used

#---------------------------------------------------------------------------------------#
def Cont_Levy_Flight(pop,lb,ub,varID,S):
    """
    Generate new children from a current population using Levy flights according to the Mantegna algorithm. 
    Applies rejection boundaries to ensure all solutions lie within the design space.
   
    Parameters
    ==========
    pop : list of arrays
        The current parent sets of design variables representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    varID : array
        A truth array indicating the location of the variables to be permuted
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Returns
    =======
    children : list of arrays
        The proposed children sets of design variables representing new system designs
    used : list
        A list of the identities of the chosen index for each child
    """
    
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Levy_Flight function.'
    assert len(lb)==len(pop[0]), 'Bounds and pop have different #s of design variables in Levy_Flight function.' 
    assert len(lb)==len(varID), 'The bounds size ({}) must be consistent with the size of the variable ID truth vectorin Levy_Flight function.'.format(len(lb),len(varID))
    assert S.fl>=0 and S.fl <=1, 'The probability that a Levy flight is performed must exist on (0,1]'
    
    children=[] # Local copy of children generated
            
    # Determine step size using Levy Flight
    step=Levy(len(pop[0]),len(pop),alpha=S.a,gamma=S.g,n=S.n) 
    
    # Perform global search from fl*p parents
    used=[]
    for i in range(int(S.fl*S.p)):
        k=int(np.random.rand()*S.p)
        while k in used:
            k=int(np.random.rand()*S.p)
        used.append(k)
        children.append(cp.deepcopy(pop[k])) 
        #print "pre:", children[i][3:]
        
        #Calculate Levy flight
        stepsize=1.0/S.sf*step[k,:]*varID
        #print "step:", stepsize[3:]
        children[i]=children[i]+stepsize 
        #print "post:", children[i][3:], "\n"
        
        #Build child applying variable boundaries 
        children[i]=Rejection_Bounds(pop[k],children[i],stepsize,lb,ub,S) 
        
    return children, used

#---------------------------------------------------------------------------------------#
def ScatterSearch(pop,lb,ub,varID,S,intDiscID=[]):  
    """
    Generate new designs by using inver-over on combinatorial variables.  Adapted from ideas in
    Tao. "Iver-over Operator for the TSP"
   
    Parameters
    ==========
    pop : list of arrays
        The current parent sets of design variables representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    varID : array
        A truth array indicating the location of the variables to be permuted
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
    intDiscID : array
        A truth array indicating the location of the discrete and integer variables to be permuted
   
    Returns
    =======
    children : list of arrays
        The proposed children sets of design variables representing new system designs
    used : list
        A list of the identities of the chosen index for each child
    """
    
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Cont_Crossover function.'
    assert len(lb)==len(pop[0]), 'Bounds and pop have different #s of design variables in Cont_Crossover function.' 
    assert len(lb)==len(varID), 'The bounds size ({}) must be consistent with the size of the variable ID truth vector in Crossover function.'.format(len(lb),len(varID))
    
    # If no discretes variables exist, set ID array to zero
    if len(intDiscID) != len(varID):
        intDiscID=np.zeros_like(varID)
        
    # Use scatter search to generate candidate children solutions    
    children=[]
    used=[]
    for i in range(0,int(len(pop)*S.fe),1):        
        #Randomly choose starting parent #2  
        j=int(np.random.rand()*len(pop))     
        while j==i or j in used:
            j=int(np.random.rand()*len(pop))
        used.append(i)
       
        #print "pre1:", pop[i][3:]
        #print "pre2:", pop[j][3:]
        d=(pop[j]-pop[i])/2.0
        if i<j:
            alpha=1
        else:
            alpha=-1
        beta=(abs(j-i)-1)/(len(pop)-2)
        c1=pop[i]-d*(1+alpha*beta)
        c2=pop[i]+d*(1-alpha*beta)
        tmp=c1+(c2-c1)*np.random.rand(len(pop[i]))
        #print "step:", tmp[3:]
        
        # Enforce integer and discrete constraints, if present
        tmp=tmp*varID+np.round(tmp*intDiscID)
        #print "tmp:", tmp[3:]
        
        #Build child applying variable boundaries 
        children.append(Simple_Bounds(tmp,lb,ub))
        #print "post:", children[i][3:], "\n"
            
    return children, used

#---------------------------------------------------------------------------------------#
def ScatterSearch2(funct,pop,lb,ub,varID,timeline,S,discreteID=[],discreteMap=[[]],intDiscID=[]):  
    """
    Generate new designs by using inver-over on combinatorial variables.  Adapted from ideas in
    Tao. "Iver-over Operator for the TSP"
   
    Parameters
    ==========
    pop : list of arrays
        The current parent sets of design variables representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    varID : array
        A truth array indicating the location of the variables to be permuted
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
    intDiscID : array
        A truth array indicating the location of the discrete and integer variables to be permuted
   
    Returns
    =======
    children : list of arrays
        The proposed children sets of design variables representing new system designs
    """
    
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Cont_Crossover function.'
    assert len(lb)==len(pop[0].d), 'Bounds and pop have different #s of design variables in Cont_Crossover function.' 
    assert len(lb)==len(varID), 'The bounds size ({}) must be consistent with the size of the variable ID truth vector in Crossover function.'.format(len(lb),len(varID))
    
    # If no discretes variables exist, set ID array to zero
    if len(intDiscID) != len(varID):
        intDiscID=np.zeros_like(varID)
         
    # Chose subpopulation and sort according to fitness
    dim=3
    perm=(np.random.permutation(range(S.p-dim))+dim)
    ind=[i for i in range(dim)]+[perm[i] for i in range(dim)]
    parents=[pop[i].d for i in ind]    
    ind, parents = (list(t) for t in zip(*sorted(zip(ind, parents))))
    
    # Determine the number of permutations and each index of the poulation to create the permutation
    nComb=[i for i in combinations(range(dim*2), 2)]
    ind1=np.array([i[0] for i in nComb])
    ind2=np.array([i[1] for i in nComb])
    mapComb=[i for i in combinations(ind, 2)]
    map1=np.array([i[0] for i in mapComb])
    map2=np.array([i[1] for i in mapComb])
    childToParentMap=np.concatenate((map1,map2),axis=0)    
    
    # Weight according to the relative difference between solution quality
    weight=0.75*((ind2-ind1)-1)/(len(parents)-2)+1
    weight=np.array([weight,]*3).transpose()
    
    # Check to ensure that there is sufficient diversity
    p1=np.array([parents[i] for i in ind1])
    p2=np.array([parents[i] for i in ind2])
    denom=np.maximum(p1,p2)
    delta=(p1-p2)/denom
    
    #!!!!!!!
    if delta.all()<0.001:
        import SamplingMethods as sm
        pop[-1].d=sm.Initial_Samples(lb,ub,'random',1)[0]
        pop[-1].f=1E99
        pop[-1].c=0
        (tmp,ch,timeline)=Get_Best(funct,[pop[-1]],[pop[-1].d],lb,ub,timeline,S,0,discreteID=discreteID, discreteMap=discreteMap)
        pop[-1].f=tmp[0].f
    # Calculate the step and individual vectors
    step=weight*(p1-p2)/1.5
    v1=p1-step
    v2=p2-step
    v3=2*p2-p1-step
    
    for i in range(len(v1)):
        for j in range(len(v1[i])):
            rand1=np.random.rand()
            rand3=np.random.rand()
            if v1[i,j]<lb[j] and rand1 < 0.5:
                v1[i,j]=lb[j]
            if v1[i,j]<ub[j] and rand1 < 0.5:
                v1[i,j]=ub[j]
            if v3[i,j]<lb[j] and rand3 < 0.5:
                v3[i,j]=lb[j]
            if v3[i,j]<ub[j] and rand3 < 0.5:
                v3[i,j]=ub[j]
    
    # Create candidate children
    c1=v1+(v2-v1)*np.random.rand(len(ind1),len(lb))
    c2=v2+(v3-v2)*np.random.rand(len(ind1),len(lb))
    children=np.concatenate((c1,c2),axis=0)
        
    # Use scatter search to generate candidate children solutions
    changes=0
    for i in range(0,len(children)):        
        # Enforce integer and discrete constraints, if present
        children[i]=children[i]*varID+np.round(children[i]*intDiscID)
        
        #Build child applying variable boundaries 
        children[i]=Simple_Bounds(children[i],lb,ub)
        
        (tmp,ch,timeline)=Get_Best(funct,[cp.deepcopy(pop[childToParentMap[i]])],[children[i]],lb,ub, timeline,S,0,discreteID=discreteID, discreteMap=discreteMap)
        if ch>0:
            pop[childToParentMap[i]]=cp.deepcopy(tmp[0])
            changes+=ch
            
    return children, changes, timeline

#---------------------------------------------------------------------------------------#
def Crossover(pop,S):  
    """
    Generate new designs by using inver-over on combinatorial variables.  Adapted from ideas in
    Tao. "Iver-over Operator for the TSP"
   
    Parameters
    ==========
    pop : list of arrays
        The current parent sets of design variables representing system designs
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Returns
    =======
    children : list of arrays
        The proposed children sets of design variables representing new system designs
    """

    children=[]
        
    feval=0
    for i in range(0,int(len(pop)*S.fe),1):        
        #Randomly choose starting parent #2  
        rand=int(np.random.rand()*len(pop))     
        while rand==i:
            rand=int(np.random.rand()*len(pop))
        
        # Randomly choose crossover point
        r=int(np.random.rand()*len(pop[i]))
        if np.random.rand()<0.5:
            children.append(np.array(pop[i][0:r+1].tolist()+pop[rand][r+1:].tolist()))
        else:
            children.append(np.array(pop[rand][0:r+1].tolist()+pop[i][r+1:].tolist()))
            
    return children

#---------------------------------------------------------------------------------------#
def Cont_Crossover(pop,lb,ub,varID,S,intDiscID=[]):  
    """
    Generate new children using distance based crossover strategies on the top parent.  
    Ideas adapted from Walton "Modified Cuckoo Search: A New Gradient Free Optimisation Algorithm" 
    and Storn "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces"
   
    Parameters
    ==========
    pop : list of arrays
        The current parent sets of design variables representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    varID : array
        A truth array indicating the location of the variables to be permuted
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
    intDiscID : array
        A truth array indicating the location of the discrete and integer variables to be permuted
   
    Returns
    =======
    children : list of arrays
        The proposed children sets of design variables representing new system designs
    used : list
        A list of the identities of the chosen index for each child
    """
    
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Cont_Crossover function.'
    assert len(lb)==len(pop[0]), 'Bounds and pop have different #s of design variables in Cont_Crossover function.' 
    assert len(lb)==len(varID), 'The bounds size ({}) must be consistent with the size of the variable ID truth vector in Cont_Crossover function.'.format(len(lb),len(varID))
    assert 0<=S.fe<=1, 'fe must be between 0 and 1 inclusive in Elite_Crossover function.'
    
    # If no discretes variables exist, set ID array to zero
    if len(intDiscID) != len(varID):
        intDiscID=np.zeros_like(varID)
    
    # Initialize variables
    golden_ratio=(1.+m.sqrt(5))/2.  # Used to bias distance based mutation strategies
    feval=0
    dx=np.zeros_like(pop[0])
    
    # Crossover top parent with an elite parent to speed local convergence
    children=[]
    used=[]
    for i in range(0,int(S.fe*len(pop)),1):
        r=int(np.random.rand()*S.p)
        while r in used or r==i:
            r=int(np.random.rand()*S.p)
        used.append(i)
        
        children.append(cp.deepcopy(pop[r]))
        #print "pre:", children[i][3:]
        dx=abs(pop[i]-children[i])/golden_ratio
        #print "dx:", dx[3:]
        children[i]=children[i]+dx*varID+np.round(dx*intDiscID)
        #print "tmp:", children[i][3:]
        children[i]=Simple_Bounds(children[i],lb,ub)
        #print "post:", children[i][3:], "\n"
        
    return children, used

#---------------------------------------------------------------------------------------#               
def Mutate(pop,lb,ub,varID,S,intDiscID=[]):
    """
    Generate new children by adding a weighted difference between two population vectors
    to a third vector.  Ideas adapted from Storn, "Differential Evolution - A Simple and
    Efficient Heuristic for Global Optimization over Continuous Spaces" and Yang, "Nature
    Inspired Optimmization Algorithms"
    
    Parameters
    ==========
    pop : list of arrays
        The current parent sets of design variables representing system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    varID : array
        A truth array indicating the location of the variables to be permuted
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========   
    intDiscID : array
        A truth array indicating the location of the discrete and integer variables to be permuted
    
    Returns
    =======
    children : list of arrays
        The proposed children sets of design variables representing new system designs
    """
    
    assert len(pop[0])==len(lb), 'Pop and best have different #s of design variables in Mutate function.'
    assert len(lb)==len(ub), 'Lower and upper bounds have different #s of design variables in Mutate function.'
    assert len(lb)==len(varID), 'The bounds size ({}) must be consistent with the size of the variable ID truth vector in Mutate function.'.format(len(lb),len(varID))
    assert S.fd>=0 and S.fd <=1, 'The probability that a pop is discovered must exist on (0,1]'
    
    children=[]
            
    # If no discretes variables exist, set ID array to zero
    if len(intDiscID) != len(varID):
        intDiscID=np.zeros_like(varID)
        
    #Discover (1-fd); K is a status vector to see if discovered
    K=np.random.rand(len(pop),len(pop[0]))>S.fd
        
    #Bias the discovery to the worst fitness solutions
    childn1=cp.copy(np.random.permutation(pop))
    childn2=cp.copy(np.random.permutation(pop)) 
        
    #New solution by biased/selective random walks
    r=np.random.rand()
    for j in range(0,len(pop),1):
        n=np.array((childn1[j]-childn2[j]))
        #print "pre:", pop[j][3:]
        step_size=r*n*varID+(n*intDiscID).astype(int)
        #print "step:", step_size[3:]
        tmp=(pop[j]+step_size*K[j,:])*varID+(pop[j]+step_size*K[j,:])*intDiscID%(ub+1-lb)
        #print "tmp", tmp[3:]
        children.append(Simple_Bounds(tmp,lb,ub))
        #print "children", tmp[3:], "\n"
        
    return children  
