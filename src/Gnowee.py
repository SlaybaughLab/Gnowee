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

@date 13May17
"""

import time

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
    initNum = min(initNum, len(initParams))
    for p in range(0, initNum, 1):
        pop.append(Parent(fitness=1E99, variables=initParams[p]))

    # Calculate initial fitness values and trim population to gh.population
    (pop, changes, timeline) = gh.population_update(pop,
                                                 [p.variables for p in pop],
                                                 timeline=timeline,
                                                 genUpdate=1)
    if len(pop) > gh.population:
        pop = pop[0:gh.population]
    else:
        gh.population = len(pop)

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
        if sum(gh.iID)+sum(gh.dID) >= 1 and sum(gh.cID) >= 1:
            (dChildren, dind) = gh.disc_levy_flight([p.variables for p in pop])
            (cChildren, cind) = gh.cont_levy_flight([p.variables for p in pop])
            children = []
            ind = []
            for i in range(0, len(cind)):
                if cind[i] in dind:
                    t = dind.index(cind[i])
                    children.append(dChildren[t]*(gh.iID+gh.dID) \
                                    +cChildren[i]*gh.cID)
                    ind.append(cind[i])
                    del dChildren[t]
                    del dind[t]
                else:
                    children.append(cChildren[i])
                    ind.append(cind[i])
            for i in range(len(dind)):
                children.append(dChildren[i])
                ind.append(dind[i])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                      timeline=timeline,
                                                      adoptedParents=ind,
                                                      mhFrac=0.2)

        elif sum(gh.cID) >= 1 and sum(gh.iID)+sum(gh.dID) == 0:
            (children, ind) = gh.cont_levy_flight([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                      timeline=timeline,
                                                      adoptedParents=ind,
                                                      mhFrac=0.2,
                                                      randomParents=True)

        elif sum(gh.cID) == 0 and sum(gh.iID)+sum(gh.dID) >= 1:
            (children, ind) = gh.disc_levy_flight([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                     timeline=timeline,
                                                     adoptedParents=ind,
                                                     mhFrac=0.2,
                                                     randomParents=True)

        # Crossover
        if sum(gh.cID)+sum(gh.iID)+sum(gh.dID) >= 1:
            (children, ind) = gh.crossover([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                     timeline=timeline)

        # Scatter Search
        if sum(gh.cID)+sum(gh.iID)+sum(gh.dID) >= 1:
            (children, ind) = gh.scatter_search([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                     timeline=timeline,
                                                     adoptedParents=ind)

        # Elite Crossover
        if sum(gh.cID)+sum(gh.iID)+sum(gh.dID) >= 1:
            children = gh.elite_crossover([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                    timeline=timeline)

        # Mutation
        if sum(gh.cID)+sum(gh.iID)+sum(gh.dID) >= 1:
            children = gh.mutate([p.variables for p in pop])
            (pop, changes, timeline) = gh.population_update(pop, children,
                                                    timeline=timeline,
                                                    genUpdate=1)

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

    #Determine execution time
    print "Program execution time was {}.".format(time.time() - startTime)
    return timeline

if __name__ == '__main__':
    main()
