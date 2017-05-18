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

@date 16May17
"""

import time

import numpy as np

from numpy.random import rand

from GnoweeUtilities import Parent

#------------------------------------------------------------------------------#
def main(gh):
    """!
    @ingroup Gnowee
    Main controller program for the Gnowee optimization.

    @param gh: <em> GnoweeHeuristic object </em> \n
        An object constaining the problem definition and the settings and
        methods required for the Gnowee optimization algorithm. \n

    @return \e list: List for design event objects for the current top solution
        vs generation. Only stores the information when new optimal designs are
        found. \n
    """

    startTime = time.time()     #Start Clock
    timeline = []                #List of history objects
    pop = []                     #List of parent objects

    # Check for objective function(s)
    assert hasattr(gh.objective.func, '__call__'), ('Invalid function '
                                                    'handle provided.')

    # Initialize population with random initial solutions
    initNum = max(gh.population*2, len(gh.ub)*10)
    initParams = gh.initialize(initNum, gh.initSampling)

    # Build the population
    initNum = min(initNum, len(initParams))
    for p in range(0, initNum, 1):
        pop.append(Parent(fitness=1E99, variables=initParams[p]))

    # Calculate initial fitness values and trim population to gh.population
    (pop, changes, timeline) = gh.population_update(pop,
                                                 [p.variables for p in pop],
                                                 timeline=timeline)
    if len(pop) > gh.population:
        pop = pop[0:gh.population]
    else:
        gh.population = len(pop)

    # Set initial heuristic probabilities
    fm = gh.fracMutation
    fe = gh.fracElite
    fl = gh.fracLevy

    # Iterate until termination criterion met
    converge = False
    while converge == False:

        # Sample generational heuristic probabilities
        gh.fracMutation = rand()*fm
        gh.fracElite = rand()*fe
        gh.fracLevy = rand()*fl

        # 3-Opt
        if sum(gh.xID) >= 1:
            (children, ind) = gh.three_opt([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                    timeline=timeline,
                                                     adoptedParents=ind)

        # Levy flights
        if sum(gh.iID)+sum(gh.dID) >= 1:
            (dChildren, dind) = gh.disc_levy_flight([p.variables for p in pop])
        else:
            dind = []
        if sum(gh.cID) >= 1:
            (cChildren, cind) = gh.cont_levy_flight([p.variables for p in pop])
        else:
            cind = []
        if sum(gh.xID) >= 1:
            (xChildren, xind) = gh.comb_levy_flight([p.variables for p in pop])
        else:
            xind = []

        # Combine the results to a single population
        children, ind = ([] for i in range(2))
        ind = list(set(cind + dind + xind))
        for i in range(0, len(ind)):
            d = np.zeros_like(gh.ub)
            c = np.zeros_like(gh.ub)
            x = np.zeros_like(gh.ub)
            if ind[i] in dind:
                d = dChildren[dind.index(ind[i])]
            if ind[i] in cind:
                c = cChildren[cind.index(ind[i])]
            if ind[i] in xind:
                x = xChildren[xind.index(ind[i])]

            # Identify the levy types used in this solution
            tmpID = 0
            if sum(d) != 0:
                tmpID += gh.iID+gh.dID
            if sum(c) != 0:
                tmpID += gh.cID
            if sum(x) != 0:
                tmpID += gh.xID
            tmp = (d+c+x)*abs((tmpID)-np.ones_like(tmpID))
            children.append(tmp+(d*(gh.iID+gh.dID)+c*gh.cID+x*gh.xID))

        (pop, changes, timeline) = gh.population_update(pop, children,
                                                      timeline=timeline,
                                                      adoptedParents=ind,
                                                      mhFrac=0.2)

        # Crossover
        if sum(gh.cID + gh.iID + gh.dID) >= 1:
            (children, ind) = gh.crossover([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                     timeline=timeline)

        # Scatter Search
        if sum(gh.cID + gh.iID + gh.dID) >= 1:
            (children, ind) = gh.scatter_search([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                     timeline=timeline,
                                                     adoptedParents=ind)

        # Mutation
        if sum(gh.cID + gh.iID + gh.dID) >= 1:
            children = gh.mutate([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                    timeline=timeline)

        # Inversion Crossover
        if sum(gh.cID + gh.iID + gh.dID + gh.xID) >= 1:
            (children, ind) = gh.inversion_crossover([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                            timeline=timeline,
                                                            adoptedParents=ind)

        # 2-Opt
        if sum(gh.xID) >= 1:
            (children, ind) = gh.two_opt([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                    timeline=timeline,
                                                     adoptedParents=ind)

        # Test generational and function evaluation convergence
        if timeline[-1].evaluations > gh.stallLimit:
            if timeline[-1].evaluations > \
               timeline[-2].evaluations+gh.stallLimit:
                converge = True
                print "Stall at evaluation #{}".format(
                    timeline[-1].evaluations)
        elif timeline[-1].generation > gh.maxGens:
            converge = True
            print "Max generations reached."
        elif timeline[-1].evaluations > gh.maxFevals:
            converge = True
            print "Max function evaluations reached."

        # Test fitness convergence
        if gh.optimum == 0.0:
            if timeline[-1].fitness < gh.optConvTol:
                converge = True
                print "Fitness Convergence"
        elif abs((timeline[-1].fitness-gh.optimum)/gh.optimum) \
              <= gh.optConvTol:
            converge = True
            print "Fitness Convergence"
        elif timeline[-1].fitness < gh.optimum:
            converge = True
            print "Fitness Convergence"

        # Update Timeline
        timeline[-1].generation += 1

    #Determine execution time
    print "Program execution time was {}.".format(time.time() - startTime)
    return timeline

if __name__ == '__main__':
    main()
