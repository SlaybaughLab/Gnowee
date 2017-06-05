"""!
@file src/Gnowee.py
@package Gnowee

@defgroup Gnowee Gnowee

@brief Main program for the Gnowee metaheuristic algorithm.

@version 1.0

Gnowee is a general purpose hybrid metaheuristic optimization algorithm
designed for rapid convergence to nearly globally optimum solutions for complex,
constrained engineering problems with mixed-integer and combinatorial design
vectors and high-cost, noisy, discontinuous, black box objective function
evaluations. Gnowee's hybrid metaheuristic framework is based on a set of
diverse, robust heuristics that appropriately balance diversification and
intensification strategies across a wide range of optimization problems.

Comparisons between Gnowee and several well-established metaheuristic
algorithms are made for a set of eighteen continuous, mixed-integer,
and combinatorial benchmarks. A summary of these benchmarks is
<a href='../../Benchmarks/results/Gnowee_Benchmark_Results.pdf'>available</a>.
These results demonstrate Gnoweee to have superior flexibility and convergence
characteristics over a wide range of design spaces.

A paper, describing the Gnowee framework and benchmarks is
<a href='../IEEE_Gnowee.pdf'>available</a>.

For examples on how to run Gnowee, please refer to the
<a href='runGnowee.ipynb'>runGnowee ipython notebook </a> included in
the <a href='../../src/'>src directory</a>.

@author James Bevins

@date 23May17

@copyright <a href='../licensing/COPYRIGHT'>&copy; 2017 UC Berkeley Copyright
            and Disclaimer Notice</a>
@license <a href='../licensing/LICENSE'>GNU GPLv3.0+ </a>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
    fe = gh.fracElite
    fl = gh.fracLevy

    # Iterate until termination criterion met
    converge = False
    while converge == False:

        # Sample generational heuristic probabilities if MI problem
        if sum(gh.iID)+sum(gh.dID) >= 1:
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
                                                        mhFrac=0.2,
                                                        randomParents=True)

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
        if sum(gh.dID + gh.xID) >= 1:
            (children, ind) = gh.inversion_crossover(
                                                    [p.variables for p in pop])
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
        if timeline[-1].generation > gh.maxGens:
            converge = True
            print "Max generations reached."
        if timeline[-1].evaluations > gh.maxFevals:
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
