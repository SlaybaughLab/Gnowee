#######################################################################################################
#
# Module : GnoweeUtilities.py
#
# Contains : Functions and methods useful across the suite of Gnowee algorithms.
#
# Author : James Bevins
#
# Last Modified: 18Apr17
#
#######################################################################################################

import numpy as np
import copy as cp
import bisect

import GnoweeHeuristics as gh#import Initialize

#---------------------------------------------------------------------------------------# 
class Parent:
    """
    Creates an object representing a current design and associated parameters
   
    Attributes
    ==========
    fitness : float
        The assessed design fitness
    design_variables : array
        The variables representing the design solution 
        [Used for continuous or discrete optimization]
    changes : integer
         The number of improvements found to the current population member
    stall : integer
         The number of evals since last improvement
    Returns
    =======
    None
    """
        
    def __init__(self,fitness=1E15,design_variables=[],changes=0,stall=0):
        self.f=fitness
        self.d=design_variables
        self.c=changes
        self.s=stall
        
        
#---------------------------------------------------------------------------------------#  
class Event:
    """
    Creates an event object representing a snapshot in the optimization process
   
    Attributes
    ==========
    generation : integer
        The generation the design was arrived at
    evaluations : integer
        The number of fitness evaluations done to obtain this design
    fitness : scalar
        The assessed design fitness
    design : array
        The variables representing the design solution
    Returns
    =======
    None
    """
        
    def __init__(self,generation,evaluations,fitness,design):
        self.g=generation
        self.e=evaluations
        self.f=fitness
        self.d=design   
        
#---------------------------------------------------------------------------------------#    
class Settings:
    """
    Creates a object representing the settings for the optimization algorithm
   
    Attributes
    ==========
    population : int
        The number of members in each generation (Default: 25)
    initial_sampling : string
        The method used to sample the phase space and create the initial population (Default: 'random')
        Valid('random','nolh','nolh-rp','nolh-cdr',and 'lhc') 
    frac_discovered : scalar
        Discovery probability (Default: 0.25)
    frac_elite : scalar
        Elite fraction (Default: 0.2)
    max_gens : int
        The maximum number of generations to search (Default: 10000)
    feval_max : int
        The maximum number of objective function evaluations (Default: 100000)
    conv_tol : scalar
        The minimum change of the best objective value before the search
        terminates (Default: 1e-5)
    stall_iter_limit : int
        The maximum number of genrations to search without a descrease
        exceeding conv_tol (Default: 200)       
    optimal_fitness : scalar
        The best know fitness value for the problem considered (Default: 0)
    opt_conv_tol : scalar
        The maximum deviation from the best know fitness value before the search
        terminates (Default: 1e-2) 
    alpha : scalar
        Levy exponent - defines the index of the distribution and controls scale properties of the stochastic process
        (Default: 1.5)
    gamma : scalar
        Gamma - Scale unit of process for Levy flights (Default: 1)
    n : integer
        Number of independent variables - can be used to reduce Levy flight variance (Default: 1)
    scaling_factor : scalar
        Step size scaling factor used to adjust Levy flights to length scale of system (Default: 10)
    penalty : scalar
        Individual constraint violation penalty to objective function (Default: 0.0)
    Returns
    =======
    None
    """
        
    def __init__(self,population=25,initial_sampling='random',frac_discovered=0.25,frac_elite=0.2, frac_levy=0.4,
                 max_gens=20000, feval_max=200000,conv_tol=1e-6,stall_iter_limit=400,optimal_fitness=0,
                 opt_conv_tol=1e-2,alpha=1.5, gamma=1,n=1,scaling_factor=10.0, penalty=0.0):          
        self.p=population
        self.s=initial_sampling           
        self.fd=frac_discovered 
        self.fe=frac_elite         
        self.fl=frac_levy
        self.gm=max_gens            
        self.em=feval_max       
        self.ct=conv_tol              
        self.sl=stall_iter_limit         
        self.of=optimal_fitness        
        self.ot=opt_conv_tol             
        self.a=alpha                    
        self.g=gamma                  
        self.n=n                        
        self.sf=scaling_factor  
        self.pen=penalty
        
#---------------------------------------------------------------------------------------#
def Get_Best(func,parents,children,lb,ub,varType,timeline,S,genUpdate,indices=[],mhFrac=0.0,random_replace=False,discreteID=[],discreteMap=[[]]):
    """
    Calculate fitness and find the current best design
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    parents : list of parent objects
        The current parents representing system designs
    children : list of arrays
        The children design variables representing new system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    varType : array
        The type of variable for each design parameter.
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
    S : Object    
        An object representing the settings for the optimization algorithm
    genUpdate : int
        Indicator for how many generations to incrment the counter by.  Genenerally 0 or 1.
   
    Optional
    ========
    mhFrac : float
        The Metropolis-Hastings fraction.  A fraction of the otherwise discarded solutions will be evaluated 
        for acceptance. 
    random_replace : boolean
        If True, a random parent will be selected for comparison to the ith child.
        (Default: True)   
    discreteID : array
        A truth array indicating the location of the discrete variables. Used to save the actual value 
        instead of index if discret variables are present.
        Default=[]
    discreteMap : list of list(s)
        nxm with n=# of discrete variables and m=# of values that can be taken for each variable 
        Default=[[]]
   
    Returns
    =======
    parents : list of parent objects
        The current parents representing system designs
    replace : integer
        The number of replacements made
    timeline : list of history objects
        The updated history of the optimization process containing best design, fitness, 
        generation, and function evaluations
    """
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    if discreteID!=[]:
        assert np.sum(discreteID)==len(discreteMap), 'A map must exist for each discrete variable. {} discrete variables, \
        but {} maps provided.'.format(np.sum(discreteID),len(discreteMap))
    
    # Track # of replacements to track effectiveness of search methods
    replace=0
    
    # Find worst fitness to use as the penalty
    for i in range(0,len(children),1):
        (fnew,gnew)=func(children[i],penalty=0)
        if fnew > S.pen:
            S.pen=fnew
    # Calculate fitness; replace parents if child has better fitness
    feval=0
    for i in range(0,len(children),1):
        if random_replace:
            j=int(np.random.rand()*len(parents))
        elif len(indices)==len(children):
            j=indices[i]
        else: 
            j=i
        (fnew,gnew)=func(children[i],penalty=S.pen)
        feval += 1
        if fnew < parents[j].f:
            parents[j].f=fnew
            parents[j].d=cp.copy(children[i])
            parents[i].c+=1
            parents[i].s=0
            replace+=1
            if parents[i].c>=25 and j>=S.p*S.fe:
                #print "Max Changes: ", parents[i].f
                parents[i].d=gh.Initialize(1, 'random', lb, ub, varType).flatten()
                (fnew,gnew)=func(parents[i].d,penalty=S.pen)
                parents[i].f=fnew
                parents[i].c=0
        else:
            parents[j].s+=1
            if parents[j].s>50000 and j!=0:
                parents[i].d=gh.Initialize(1, 'random', lb, ub, varType).flatten()
                (fnew,gnew)=func(parents[i].d,penalty=S.pen)
                parents[i].f=fnew
                parents[i].c=0
                parents[i].s=0
                       
            # Metropis-Hastings algorithm
            r=int(np.random.rand()*len(parents))
            if r<=mhFrac:
                r=int(np.random.rand()*len(parents))
                if fnew < parents[r].f:  
                    parents[r].f=fnew
                    parents[r].d=cp.copy(children[i])
                    parents[r].c+=1
                    parents[r].s+=1
                    replace+=1

    #Sort the pop
    parents.sort(key=lambda x: x.f)
    
    # Map the discrete variables for storage
    if discreteID!=[]:
        dVec=[]
        i=0
        for j in range(len(discreteID)):
            if discreteID[j]==1:
                dVec.append(discreteMap[i][int(parents[0].d[j])])
                i+=1
            else:
                dVec.append(parents[0].d[j])
    else:
        dVec=cp.copy(parents[0].d)
        
    #Store history on timeline if new optimal design found
    if len(timeline)<2:
        timeline.append(Event(len(timeline)+1,feval,parents[0].f,dVec))
    elif parents[0].f<timeline[-1].f and abs((timeline[-1].f-parents[0].f)/parents[0].f) > S.ct:
        timeline.append(Event(timeline[-1].g+1,timeline[-1].e+feval,parents[0].f,dVec))
    else:
        timeline[-1].g+=genUpdate
        timeline[-1].e+=feval
        
    return(parents,replace,timeline)

#---------------------------------------------------------------------------------------#
def Simple_Bounds(tmp,lb,ub):
    """
    Application of problem boundaries to generated solutions
   
    Parameters
    ==========
    tmp : array
        The proposed new system designs
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Returns
    =======
    tmp : array
        The new system designs that are within problem boundaries
    """
    
    assert len(tmp)==len(lb), 'tmp and lb best have different # of design variables in Simple_Bounds function.'
    assert len(ub)==len(lb), 'Boundaries best have different # of design variables in Simple_Bounds function.'
            
    #Apply lower bound
    for i in range(0,len(tmp),1):        
        if tmp[i]<lb[i]:
            tmp[i]=lb[i]

    #Apply upper bound
    for i in range(0,len(tmp),1):
        if tmp[i]>ub[i]:
            tmp[i]=ub[i]
  
    return tmp

#---------------------------------------------------------------------------------------# 
def Rejection_Bounds(parent,child,stepsize,lb,ub,S):
    """
    Application of problem boundaries to generated solutions. Adjusts step size for all rejected
    solutions until within the boundaries.  
   
    Parameters
    ==========
    parent : array
        The current system designs
    child : array
        The proposed new system designs
    stepsize : float
        The Levy flight stepsize
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    S : Gnowee Settings Object    
        An object representing the settings for the Gnowee optimization algorithm
   
    Returns
    =======
    child : array
        The new system design that are within problem boundaries
    """
    
    assert len(child)==len(lb), 'Child and lb best have different # of design variables in Rejection_Bounds function.'
    assert len(ub)==len(lb), 'Boundaries best have different # of design variables in Rejection_Bounds function.'
            
    for i in range(0,len(child),1): 
        change_count=0
        while child[i]<lb[i] or child[i]>ub[i]:
            if change_count >= 5:
                child[i]=cp.copy(parent[i])
            else:
                stepsize[i]=stepsize[i]/2.0  
                child[i]=child[i]-stepsize[i]
                change_count+=1
    return child

#---------------------------------------------------------------------------------------#
class WeightedRandomGenerator(object):
    """
    Defines a class of weights to be used to select number of instances in array randomly with
    linear weighting.  
   
    Parameters
    ==========
    self : object
        Current instance of the class
    weights : array
        The array of weights (Higher = more likely to be selected)
   
    Returns
    =======
    bisect.bisect_right(self.totals, rnd) : integer
        The randomly selected index of the weights array
    """
    
    def __init__(self, weights):
        self.totals = []
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = np.random.rand() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()