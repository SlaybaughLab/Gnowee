#######################################################################################################
#
# Module : Utilities.py
#
# Contains : Functions and methods useful across the suite of Gnowee algorithms.
#
# Author : James Bevins
#
# Last Modified: 16Aug16
#
#######################################################################################################

import numpy as np
import copy as cp
import bisect

#---------------------------------------------------------------------------------------# 
class Parent:
    """
    Creates a parent object representing a current design
   
    Attributes
    ==========
    fitness : scalar
        The assessed design fitness
    design_variables : array
        The variables representing the design solution 
        [Used for continuous or discrete optimization]
    index : integer
        Represents the starting index for the next generation, which is the last city visited.  (Default=0)
        [Used for TSP problems, ZhouDCS in particular.]
    Returns
    =======
    None
    """
        
    def __init__(self,fitness=1E15,design_variables=[],index=0):
        self.f=fitness
        self.d=design_variables
        self.i=index
        
        
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
        Step size scaling factor used to adjust Levy flights to length scale of system (Default: 100)
    step_size : scalar
        Step size parameter used for generational cooling (Default: 1.0)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    Returns
    =======
    None
    """
        
    def __init__(self,population=25,initial_sampling='random',frac_discovered=0.25,frac_elite=0.2, frac_levy=0.4,
                 max_gens=20000, feval_max=200000,conv_tol=1e-6,stall_iter_limit=400,optimal_fitness=0,
                 opt_conv_tol=1e-2,alpha=1.5, gamma=1,n=1,scaling_factor=100.0,debug=False):          
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
        self.d=debug                     
        
#---------------------------------------------------------------------------------------#
def Get_Best(func,parents,children,timeline,S,random_replace=False):
    """
    Calculate fitness and find the current best design
   
    Parameters
    ==========
    func : function
        The objective function to be minimized
    parents : list of parent objects
        The current parents representing system designs
    children : list of parent objects
        The children representing new system designs
    timeline : list of history objects
        The histories of the optimization process containing best design, fitness, generation, and 
        function evaluations
    S : Object    
        An object representing the settings for the optimization algorithm
   
    Optional
    ========
    random_replace : boolean
        If True, a random parent will be selected for comparison to the ith child.
        (Default: True)   
   
    Returns
    =======
    parents : list of parent objects
        The current parents representing system designs
    timeline : list of history objects
        The updated history of the optimization process containing best design, fitness, 
        generation, and function evaluations
    """
    
    #Test input values for consistency
    assert hasattr(func, '__call__'), 'Invalid function handle'
    
    # Track # of replacements to track effectiveness of search methods
    replace=0
    
    # Calculate fitness; replace parents if child has better fitness
    feval=0
    for i in range(0,len(children),1):
        if random_replace:
            j=int(np.random.rand()*len(parents))
        else:
            j=i
        (fnew,gnew)=func(children[i].d)
        feval += 1
        if fnew < parents[j].f:
            parents[j].f=fnew
            parents[j].d=cp.copy(children[i].d)
            replace+=1

    #Sort the pop
    parents.sort(key=lambda x: x.f)
    
    #Store history on timeline if new optimal design found
    if len(timeline)<2:
        timeline.append(Event(len(timeline)+1,feval,parents[0].f,parents[0].d))
    elif parents[0].f<timeline[-1].f and abs((timeline[-1].f-parents[0].f)/parents[0].f) > S.ct:
        timeline.append(Event(timeline[-1].g+1,timeline[-1].e+feval,parents[0].f,parents[0].d))
    else:
        timeline[-1].g+=1
        timeline[-1].e+=feval
  
    if S.d:
        print "\n At end of function Get_Best,"
        print "The number of replacements was %d" %replace
        print "The best parent is", timeline[-1].d
        print "With a fitness of = %f, taking %d generations and %d function evaluations to accomplish" \
               %(timeline[-1].f,timeline[-1].g,timeline[-1].e)
        print "Current parents are: " 
        for i in range(len(parents)) :    
            print "Parent ",i,": Fitness=",parents[i].f,", Design=",parents[i].d 
    return(parents,timeline)

#---------------------------------------------------------------------------------------#
def Simple_Bounds(tmp,lb,ub,debug=False,change_count=0):
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
   
    Optional
    ========
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)   
    change_count : integer
        Counter to track the number of solutions that occur outside of problem boundaries.  
        Can be used to diagnose too large or small of alpha
        (Default: 0)
   
    Returns
    =======
    tmp : array
        The new system designs that are within problem boundaries
    """
    
    assert len(tmp)==len(lb), 'S and lb best have different # of design variables in Simple_Bounds function.'
    assert len(ub)==len(lb), 'Boundaries best have different # of design variables in Simple_Bounds function.'
            
    #Apply lower bound
    for i in range(0,len(tmp),1):        
        if tmp[i]<lb[i]:
            if debug:
                print "tmp[%d] = %f, lb[%d] = %f " %(i,tmp[i],i,lb[i])
            tmp[i]=lb[i]
            change_count+=1

    #Apply upper bound
    for i in range(0,len(tmp),1):
        if tmp[i]>ub[i]:
            if debug:
                print "tmp[%d] = %f, ub[%d] = %f " %(i,tmp[i],i,ub[i])
            tmp[i]=ub[i]
            change_count+=1
    
    if debug:
        print "The number of changes were: %d" %change_count
    return tmp

#---------------------------------------------------------------------------------------# 
def Rejection_Bounds(parent,child,stepsize,lb,ub,S,debug=False,change_count=0):
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
   
    Optional
    ======== 
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)  
    change_count : integer
        Counter to track the number of solutions that occur outside of problem boundaries.  
        Can be used to diagnose too large or small of alpha
        (Default: 0)
   
    Returns
    =======
    child : array
        The new system design that are within problem boundaries
    """
    import SamplingMethods as sm
    
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
    if debug:
        print "The number of changes were: %d" %change_count
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