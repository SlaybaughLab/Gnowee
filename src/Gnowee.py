"""!
@file src/Gnowee.py
@package Gnowee

@defgroup Gnowee Gnowee

@brief Main program for the Gnowee metaheuristic algorithm.

General nearly-global metaheuristic optimization algorithm. Uses a blend of
common heuristics to solve difficult gradient free constrained MINLP problems
with categorical variables. It is capable of solving  simpler problems, but may
not be the algorithm of choice.

For examples on how to run Gnowee, please refer to the runGnowee notebook
included in the src directory.

@author James Bevins

@date 9May17
"""

import time

import numpy as np

from numpy.random import rand

#from GnoweeHeuristics import disc_levy_flight, cont_levy_flight, crossover,
#                             cont_crossover, mutate, scatterSearch, initialize
from GnoweeUtilities import Parent, Get_Best

#------------------------------------------------------------------------------#
def main(func, lb, ub, varType, gh, discreteVals=[]):
    """!
    @ingroup Gnowee
    Main controller program for the Gnowee optimization.

    @param func: \e function \n
        The objective function to be minimized. \n
    @param lb: <em> list or array </em> \n
        The lower bounds of the design variable(s). Only enter the bounds for
        continuous and integer/binary variables. The order must match the
        order specified in varType and ub. \n
    @param ub: <em> list or array </em> \n
        The upper bounds of the design variable(s). Only enter the bounds for
        continuous and integer/binary variables. The order must match the
        order specified in varType and lb. \n
    @param varType: <em> list or array </em> \n
        The type of variable for each position in the upper and lower bounds
        array. Discrete variables are to be included last as they are specified
        separatly from the lb/ub throught the discreteVals optional input.
        A variable can have two types (for example, 'dx' could denote a layer
        that can take multiple materials and be placed at multiple design
        locations) \n
        Allowed values: \n
         'c' = continuous over a given range (range specified in lb & ub). \n
         'i' = integer/binary (difference denoted by ub/lb). \n
         'd' = discrete where the allowed values are given by the option
               discreteVals nxm arrary with n=# of discrete variables and
               m=# of values that can be taken for each variable. \n
         'x' = combinatorial. All of the variables denoted by x are assumed
               to be "swappable" in combinatorial permutations.  There must be
               at least two variables denoted as combinatorial. \n
         'f' = fixed design variable. Will not be considered of any
               permutation. \n
    @param: gh: <em> GnoweeHeuristic object </em> \n
        An object constaining the settings and methods required for the
        Gnowee optimization algorithm. \n
    @param discreteVals: <em> list of list(s) </em> \n
        nxm with n=# of discrete variables and m=# of values that can be taken
        for each variable. For example, if you had two variables representing
        the tickness and diameter of a cylinder that take standard values, the
        discreteVals could be specified as: \n
        discreteVals = [[0.125, 0.25, 0.375], [0.25, 0.5, 075]] \n
        Gnowee will then map the optimization results to these allowed values.

    @return \e list: List for design event objects for the current top solution
        vs generation. Only stores the information when new optimal designs are
        found. \n
    """

    startTime = time.time()     #Start Clock
    timeline = []                #List of history objects
    pop = []                     #List of parent objects

    # Check input variables
    assert varType.count('d') == len(discreteVals), ('The allowed discrete '
                        'values must be specified for each discrete variable.'
                        '{} in varType, but {} in discreteVals.'.format(
                         varType.count('d'), len(discreteVals)))
    assert varType.count('c') + varType.count('b') + varType.count('i') \
           == len(ub), ('Each specified continuous, binary, and integer '
                        'variable must have a corresponding upper and lower '
                        'bounds. {} variables and {} bounds specified'.format(
                         varType.count('c') + varType.count('b') +
                         varType.count('i'), len(lb)))
    assert max(len(varType) - 1 - varType[::-1].index('c') \
                 if 'c' in varType else -1,
               len(varType) - 1 - varType[::-1].index('b') \
                 if 'b' in varType else -1,
               len(varType) - 1 - varType[::-1].index('i') \
                 if 'i' in varType else -1) \
                < varType.index('d'), ('The discrete variables must be '
                'specified after the continuous, binary, and integer variables.'
                ' The order given was {}'.format(varType))
    assert len(lb) == len(ub), ('The lower and upper bounds must have the same '
                'dimensions. lb = {}, ub = {}'.format(len(lb), len(ub)))
    assert set(varType).issubset(['c', 'i', 'd', 'x', 'f']), ('The variable '
                'specifications do not match the allowed values of "c", "i", or'
                ' "d".  The varTypes specified is {}'.format(varType))
    assert np.all(ub > lb), ('All upper-bound values must be greater than '
                             'lower-bound values')

    #  Append discretes to lb and ubs and convert to numpy arrays
    for d in range(len(discreteVals)):
        lb.append(0)
        ub.append(len(discreteVals[d])-1)
        discreteVals[d] = np.array(discreteVals[d])
    lb = np.array(lb)
    ub = np.array(ub)

    # Check for objective function(s)
    assert hasattr(func, '__call__'), 'Invalid function handle provided.'

    # Initialize population with random initial solutions
    initNum = max(gh.population*2, len(ub)*10)
    initParams = gh.initialize(initNum, gh.initSampling, lb, ub, varType)
    for p in range(0, initNum, 1):
        pop.append(Parent(fitness=1E99, variables=initParams[p]))

    # Develop ID vectors for each variable type
    cID = []
    iID = []
    dID = []
    xID = []
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
    cID = np.array(cID)
    iID = np.array(iID)
    dID = np.array(dID)
    xID = np.array(xID)

    # Calculate initial fitness values and trim population to gh.population
    (pop, changes, timeline) = Get_Best(func, pop, [p.variables for p in pop],
                                        lb, ub, varType, timeline, gh, 1)
    pop = pop[0:gh.population]

    # Set initial heuristic probabilities
    fd = gh.fracDiscovered
    fe = gh.fracElite
    fl = gh.fracLevy

    # Iterate until termination criterion met
    converge = False
    while timeline[-1].generation <= gh.maxGens and \
          timeline[-1].evaluations <= gh.maxFevals and converge == False:

        # Sample generational heuristic probabilities
        gh.fracDiscovered = rand()*fd
        gh.fracElite = rand()*fe
        gh.fracLevy = rand()*fl

        # Levy flights
        if sum(iID)+sum(dID) >= 1 and sum(cID) >= 1:
            (dChildren, dind) = gh.disc_levy_flight([p.variables for p in pop],
                                                     lb, ub, iID+dID)
            (cChildren, cind) = gh.cont_levy_flight([p.variables for p in pop],
                                                     lb, ub, cID)
            children = []
            ind = []
            for i in range(0, len(cind)):
                if cind[i] in dind:
                    t = dind.index(cind[i])
                    children.append(dChildren[t]*(iID+dID)+cChildren[i]*cID)
                    ind.append(cind[i])
                    del dChildren[t]
                    del dind[t]
                else:
                    children.append(cChildren[i])
                    ind.append(cind[i])
            for i in range(len(dind)):
                children.append(dChildren[i])
                ind.append(dind[i])
            (pop, changes, timeline) = Get_Best(func, pop, children, lb, ub,
                                                varType, timeline, gh, 0,
                                                indices=ind, mhFrac=0.2,
                                                discreteID=dID,
                                                discreteMap=discreteVals)

        elif sum(cID) >= 1 and sum(iID)+sum(dID) == 0:
            (children, ind) = gh.cont_levy_flight([p.variables for p in pop],
                                                  lb, ub, cID)
            (pop, changes, timeline) = Get_Best(func, pop, children, lb, ub,
                                                varType, timeline, gh, 0,
                                                indices=ind, mhFrac=0.2,
                                                random_replace=True,
                                                discreteID=dID,
                                                discreteMap=discreteVals)

        elif sum(cID) == 0 and sum(iID)+sum(dID) >= 1:
            (children, ind) = gh.disc_levy_flight([p.variables for p in pop],
                                                  lb, ub, iID+dID)
            (pop, changes, timeline) = Get_Best(func, pop, children, lb, ub,
                                                varType, timeline, gh, 0,
                                                indices=ind, mhFrac=0.2,
                                                random_replace=True,
                                                discreteID=dID,
                                                discreteMap=discreteVals)

        # Crossover
        if sum(cID)+sum(iID)+sum(dID) >= 1:
            (children, ind) = gh.crossover([p.variables for p in pop], lb, ub,
                                            cID, intDiscID=iID+dID)
            (pop, changes, timeline) = Get_Best(func, pop, children, lb, ub,
                                               varType, timeline, gh, 0,
                                               discreteID=dID,
                                               discreteMap=discreteVals)

        # Scatter Search
        if sum(cID)+sum(iID)+sum(dID) >= 1:
            (children, ind) = gh.scatter_search([p.variables for p in pop], lb,
                                                ub, cID, intDiscID=iID+dID)
            (pop, changes, timeline) = Get_Best(func, pop, children, lb, ub,
                                               varType, timeline, gh, 0,
                                               indices=ind, discreteID=dID,
                                               discreteMap=discreteVals)

        # Elite Crossover
        if sum(cID)+sum(iID)+sum(dID) >= 1:
            children = gh.elite_crossover([p.variables for p in pop])
            (pop, changes, timeline) = Get_Best(func, pop, children, lb, ub,
                                               varType, timeline, gh, 0,
                                               discreteID=dID,
                                               discreteMap=discreteVals)

        # Mutation
        if sum(cID)+sum(iID)+sum(dID) >= 1:
            children = gh.mutate([p.variables for p in pop], lb, ub, cID,
                                 intDiscID=iID+dID)
            (pop, changes, timeline) = Get_Best(func, pop, children, lb, ub,
                                                varType, timeline, gh, 1,
                                                discreteID=dID,
                                                discreteMap=discreteVals)

        # Test generational and function evaluation convergence
        if timeline[-1].generation > gh.stallLimit:
            if timeline[-1].generation > timeline[-2].generation+gh.stallLimit:
                converge = True
                print "Generational Stall at generation #{}".format(
                    timeline[-1].generation)
            elif (timeline[-2].fitness-timeline[-1].fitness) \
                  /timeline[-2].fitness < gh.convTol:
                if timeline[-1].generation > \
                      timeline[-2].generation+gh.stallLimit:
                    converge = True
                    print "Generational Convergence"
        elif timeline[-1].generation > gh.maxGens:
            converge = True
            print "Max generations reached."
        elif timeline[-1].evaluations > gh.maxFevals:
            converge = True
            print "Max function evaluations reached."

        # Test fitness convergence
        if gh.optimalFitness == 0.0:
            if timeline[-1].fitness < gh.optConvTol:
                converge = True
                print "Fitness Convergence"
        elif abs((timeline[-1].fitness-gh.optimalFitness)/gh.optimalFitness) \
              <= gh.optConvTol:
            converge = True
            print "Fitness Convergence"
        elif timeline[-1].fitness < gh.optimalFitness:
            converge = True
            print "Fitness Convergence"

    #Determine execution time
    print "Program execution time was {}.".format(time.time() - startTime)
    return timeline

if __name__ == '__main__':
    main()
