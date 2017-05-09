"""!
@file src/GnoweeUtilities.py
@package Gnowee

@defgroup GnoweeUtilities GnoweeUtilities

@brief Classes and methods to support the Gnowee optimization algorithm.

@author James Bevins

@date 8May17
"""

import numpy as np
import copy as cp
import bisect

from GnoweeHeuristics import initialize

#------------------------------------------------------------------------------#
class Parent(object):
    """!
    @ingroup GnoweeUtilities
    The class contains all of the parameters pertinent to a member of the
    population.
    """

    ##
    def __init__(self, variables=None, fitness=1E15, changeCount=0,
                 stallCount=0):
        """!
        Constructor to build the Parent class.

        @param self: <em> Parent pointer </em> \n
            The Parent pointer. \n
        @param variables: \e array \n
            The set of variables representing a design solution. \n
        @param fitness: \e float \n
            The assessed fitness for the current set of variables. \n
        @param changeCount: \e integer \n
            The number of improvements to the current population member. \n
        @param stallCount: \e integer \n
            The number of evaluations since the last improvement. \n
        """

        ##  @var variables
        # \e array
        # The set of variables representing a design solution.
        self.variables = variables

        ##  @var fitness
        # \e float
        # The assessed fitness for the current set of variables.
        self.fitness = fitness

        ##  @var changeCount
        # \e integer
        # The number of improvements to the current population member.
        self.changeCount = changeCount

        ##  @var stallCount
        # \e integer
        # he number of evaluations since the last improvement.
        self.stallCount = stallCount

    def __repr__(self):
        """!
        Parent print function.

        @param self: <em> Parent pointer </em> \n
            The Parent pointer. \n
        """
        return "Parent({}, {}, {}, {})".format(self.variables, self.fitness,
                                               self.changeCount,
                                               self.stallCount)

    def __str__(self):
        """!
        Human readable Parent print function.

        @param self: <em> Parent pointer </em> \n
            The Parent pointer. \n
        """

        header = ["Parent:"]
        header += ["Variables = {}".format(self.variables)]
        header += ["Fitness = {}".format(self.fitness)]
        header += ["Change Count = {}".format(self.changeCount)]
        header += ["Stall Count = {}".format(self.stallCount)]
        return "\n".join(header)+"\n"

#------------------------------------------------------------------------------#
class Event(object):
    """!
    @ingroup GnoweeUtilities
    Represents a snapshot in the optimization process to be used for debugging,
    benchmarking, and user feedback.
    """

    ##
    def __init__(self, generation, evaluations, fitness, design):
        """!
        Constructor to build the Event class.

        @param self: <em> Event pointer </em> \n
            The Event pointer. \n
        @param generation: \e integer \n
            The generation the design was arrived at. \n
        @param evaluations: \e integer \n
            The number of fitness evaluations done to obtain this design. \n
        @param fitness: \e float \n
            The assessed fitness for the current set of variables. \n
        @param design: \e array \n
            The set of variables representing a design solution. \n
        """

        ##  @var generation
        # \e integer
        # The generation the design was arrived at.
        self.generation = generation

        ##  @var evaluations
        # \e integer
        # The number of fitness evaluations done to obtain this design.
        self.evaluations = evaluations

        ##  @var fitness
        # \e float
        # The assessed fitness for the current set of variables.
        self.fitness = fitness

        ##  @var design
        # \e array
        # The set of variables representing a design solution.
        self.design = design

    def __repr__(self):
        """!
        Event print function.

        @param self: <em> Event pointer </em> \n
            The Event pointer. \n
        """
        return "Event({}, {}, {}, {})".format(self.generation, self.evaluations,
                                               self.fitness, self.design)

    def __str__(self):
        """!
        Human readable Event print function.

        @param self: <em> Event pointer </em> \n
            The Event pointer. \n
        """

        header = ["Event:"]
        header += ["Generation # = {}".format(self.generation)]
        header += ["Evaluation # = {}".format(self.evaluations)]
        header += ["Fitness = {}".format(self.fitness)]
        header += ["Design = {}".format(self.design)]
        return "\n".join(header)+"\n"

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
                parents[i].d=initialize(1, 'random', lb, ub, varType).flatten()
                (fnew,gnew)=func(parents[i].d,penalty=S.pen)
                parents[i].f=fnew
                parents[i].c=0
        else:
            parents[j].s+=1
            if parents[j].s>50000 and j!=0:
                parents[i].d=initialize(1, 'random', lb, ub, varType).flatten()
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