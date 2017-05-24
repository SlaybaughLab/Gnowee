"""!
@file src/GnoweeHeuristics.py
@package Gnowee

@defgroup GnoweeHeuristics GnoweeHeuristics

@brief Heuristics and settings supporting the Gnowee metaheuristic optimization
algorithm.

This instantiates the class and methods necessary to perform an optimization
using the Gnowee algorithm.  Each of the heuristics can also be used
independently of the Gnowee algorithm by instantiating this class and choosing
the desired heuristic.

The default settings are those found to be best for a suite of benchmark
problems but one may find alternative settings are useful for the problem of
interest based on the fitness landscape and type of variables.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""

import numpy as np
import copy as cp

from math import sqrt
from numpy.random import rand, permutation
from Sampling import levy, tlf, initial_samples
from GnoweeUtilities import ProblemParameters, Event

#------------------------------------------------------------------------------#
class GnoweeHeuristics(ProblemParameters):
    """!
    @ingroup GnoweeHeuristics
    The class is the foundation of the Gnowee optimization algorithm.  It sets
    the settings required for the algorithm and defines the heurstics.
    """

    ##
    def __init__(self, population=25, initSampling='lhc', fracMutation=0.2,
                 fracElite=0.2, fracLevy=1.0, alpha=0.5, gamma=1, n=1,
                 scalingFactor=10.0, penalty=0.0, maxGens=20000,
                 maxFevals=200000, convTol=1e-6, stallLimit=10000,
                 optConvTol=1e-2, **kwargs):
        """!
        Constructor to build the GnoweeHeuristics class.  This class must be
        fully instantiated to run the Gnowee program.  It consists of 2 main
        parts: The main class attributes and the inhereted ProblemParams class
        attributes.  The main class atrributes contain defaults that don't
        require direct user input to work (but can be modified by user input
        if desired), but the ProblemParameters class does require proper
        instantiation by the user.

        The default settings are found to be optimized for a wide range of
        problems, but can be changed to optimize performance for a particular
        problem type or class.  For more details, refer to the
        <a href='../docs/IEEE_Gnowee.pdf'>development paper</a>.

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param population: \e integer \n
            The number of members in each generation. \n
        @param initSampling: \e string \n
            The method used to sample the phase space and create the initial
            population. Valid options are 'random', 'nolh', 'nolh-rp',
            'nolh-cdr', and 'lhc' as specified in init_samples(). \n
        @param fracMutation : \e float \n
            Discovery probability used for the mutate() heuristic. \n
        @param fracElite: \e float \n
            Elite fraction probability used for the scatter_search(),
            crossover(), and cont_crossover() heuristics. \n
        @param fracLevy: \e float \n
            Levy flight probability used for the disc_levy_flight() and
            cont_levy_flight() heuristics. \n
        @param alpha: \e float \n
            Levy exponent - defines the index of the distribution and controls
            scale properties of the stochastic process. \n
        @param gamma: \e float \n
            Gamma - scale unit of process for Levy flights. \n
        @param n: \e integer \n
            Number of independent variables - can be used to reduce Levy flight
            sampling variance. \n
        @param penalty: \e float \n
            Individual constraint violation penalty to add to objective
            function. \n
        @param scalingFactor: \e float \n
            Step size scaling factor used to adjust Levy flights to length scale
            of system. The implementation of the Levy flight sampling makes this
            largely arbitrary. \n
        @param maxGens: \e integer \n
            The maximum number of generations to search. \n
        @param maxFevals: \e integer \n
            The maximum number of objective function evaluations. \n
        @param convTol: \e float \n
            The minimum change of the best objective value before the search
            terminates. \n
        @param stallLimit: \e integer \n
            The maximum number of evaluations to search without an
            improvement. \n
        @param optConvTol: \e float \n
            The maximum deviation from the best know fitness value before the
            search terminates. \n
        @param kwargs: <em> ProblemParameters class arguments </em> \n
            Keyword arguments for the attributes of the ProblemParameters
            class. If not provided. The inhereted attributes will be set to the
            class defaults. \n
        """

        # Initialize base ProblemParameters class
        ProblemParameters.__init__(self, **kwargs)

        ## @var population
        # \e integer:
        # The number of members in each generation.
        self.population = population

        ## @var initSampling
        # \e string:
        # The method used to sample the phase space and create the initial
        # population. Valid options are 'random', 'nolh', 'nolh-rp',
        #'nolh-cdr', and 'lhc' as specified in init_samples().
        self.initSampling = initSampling

        ## @var fracMutation
        # \e float:
        # Discovery probability used for the mutate() heuristic.
        self.fracMutation = fracMutation
        assert self.fracMutation >= 0 and self.fracMutation <= 1, ('The '
                                 'probability of discovery must exist on (0,1]')

        ## @var fracElite
        # \e float:
        # Elite fraction probability used for the scatter_search(), crossover(),
        # and cont_crossover() heuristics.
        self.fracElite = fracElite
        assert self.fracElite >= 0 and self.fracElite <= 1, ('The elitism '
                                                'fraction must exist on (0,1]')

        ## @var fracLevy
        # \e float:
        # Levy flight probability used for the disc_levy_flight() and
        # cont_levy_flight() heuristics.
        self.fracLevy = fracLevy
        assert self.fracLevy >= 0 and self.fracLevy <= 1, ('The probability '
                        'that a Levy flight is performed must exist on (0,1]')

        ## @var alpha
        # \e float:
        # Levy exponent - defines the index of the distribution and controls
        # scale properties of the stochastic process.
        self.alpha = alpha

        ## @var gamma
        # \e float:
        # Gamma - scale unit of process for Levy flights.
        self.gamma = gamma

        ## @var n
        # \e integer:
        # Number of independent variables - can be used to reduce Levy flight
        # sampling variance.
        self.n = n

        ## @var scalingFactor
        # \e float:
        # Step size scaling factor used to adjust Levy flights to length scale
        # of system. The implementation of the Levy flight sampling makes this
        # largely arbitrary.
        self.scalingFactor = scalingFactor

        ## @var penalty
        # \e float:
        # Individual constraint violation penalty to add to objective function.
        self.penalty = penalty

        ## @var maxGens
        # \e integer:
        # The maximum number of generations to search.
        self.maxGens = maxGens

        ## @var maxFevals
        # \e integer:
        # The maximum number of objective function evaluations.
        self.maxFevals = maxFevals

        ## @var convTol
        # \e float:
        # The minimum change of the best objective value before the search
        # terminates.
        self.convTol = convTol

        ## @var stallLimit
        # \e integer:
        # The maximum number of gen3rations to search without a descrease
        # exceeding convTol.
        self.stallLimit = stallLimit

        ## @var optConvTol
        # \e float:
        # The maximum deviation from the best know fitness value before the
        # search terminates.
        self.optConvTol = optConvTol

    def __repr__(self):
        """!
        GnoweeHeuristics print function.

        @param self: <em> GnoweeHeuristics pointer </em> \n
            The GnoweeHeuristics pointer. \n
        """
        return "GnoweeHeuristics({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, \
                                 {}, {}, {}, {}, {})".format(self.population,
                                                  self.initSampling,
                                                  self.fracMutation,
                                                  self.fracElite,
                                                  self.fracLevy,
                                                  self.alpha,
                                                  self.gamma,
                                                  self.n,
                                                  self.scalingFactor,
                                                  self.penalty,
                                                  self.maxGens,
                                                  self.maxFevals,
                                                  self.convTol,
                                                  self.stallLimit,
                                                  self.optConvTol,
                                                  ProblemParameters.__repr__())

    def __str__(self):
        """!
        Human readable GnoweeHeuristics print function.

        @param self: <em> GnoweeHeuristics pointer </em> \n
            The GnoweeHeuristics pointer. \n
        """
        header = ["  GnoweeHeuristics:"]
        header += ["Population = {}".format(self.population)]
        header += ["Sampling Method = {}".format(self.initSampling)]
        header += ["Discovery Fraction = {}".format(self.fracMutation)]
        header += ["Elitism Fraction = {}".format(self.fracElite)]
        header += ["Levy Fraction = {}".format(self.fracLevy)]
        header += ["Levy Alpha = {}".format(self.alpha)]
        header += ["Levy Gamma = {}".format(self.gamma)]
        header += ["Levy Independent Samples = {}".format(self.n)]
        header += ["Levy Scaling Parameter = {}".format(self.scalingFactor)]
        header += ["Constraint Violaition Penalty = {}".format(self.penalty)]
        header += ["Max # of Generations = {}".format(self.maxGens)]
        header += ["Max # of Function Evaluations = {}".format(self.maxFevals)]
        header += ["Convergence Tolerance = {}".format(self.convTol)]
        header += ["Stall Limit = {}".format(self.stallLimit)]
        header += ["Optimal Convergence Tolerance = {}".format(self.optConvTol)]
        header += ["     Attributes Inhereted from ProblemParameters:"]
        header += ["{}".format(ProblemParameters.__str__(self))]
        return "\n".join(header)+"\n"

    def initialize(self, numSamples, sampleMethod):
        """!
        Initialize the population according to the sampling method chosen.

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param numSamples: \e integer \n
            The number of samples to be generated. \n
        @param sampleMethod: \e string \n
            The method used to sample the phase space and create the initial
            population. Valid options are 'random', 'nolh', 'nolh-rp',
            'nolh-cdr', and 'lhc' as specified in init_samples(). \n

        @return <em> list of arrays: </em> The initialized set of samples.
        """

        initSamples = initial_samples(self.lb, self.ub, sampleMethod,
                                      numSamples)
        if sum(self.xID) != 0:
            xUB = [self.ub[np.where(self.xID == 1)[0][0]]]*len(self.xID)
            xSamples = initial_samples([0]*len(self.xID), xUB, "rand-wor",
                                      numSamples)

        # Map to discrete/integers
        for var in range(len(self.varType)):
            if 'i' in self.varType[var] or 'd' in self.varType[var]:
                initSamples[:, var] = np.rint(initSamples[:, var])

        # Map to combinatorial
        if sum(self.xID) != 0:
            initSamples = initSamples*(self.cID+self.iID+self.dID) \
                          +xSamples*self.xID

        return initSamples

    def disc_levy_flight(self, pop):
        """!
        Generate new children using truncated Levy flights permutation of
        current generation design parameters according to:

        \f$ L_{\alpha,\gamma}=FLOOR(TLF_{\alpha,\gamma}*D(x)), \f$

        where \f$ TLF_{\alpha,\gamma} \f$ is calculated in tlf(). Applies
        rejection_bounds() to ensure all solutions lie within the design
        space by adapting the step size to the size of the design space.

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        children = [] # Local copy of children generated
        varID = self.iID+self.dID

        # Determine step size using Levy Flight
        step = tlf(len(pop), len(pop[0]), alpha=self.alpha, gam=self.gamma)

        # Initialize Variables
        used = []
        for i in range(0, int(self.fracLevy*self.population), 1):
            k = int(rand()*self.population)
            while k in used:
                k = int(rand()*self.population)
            used.append(k)
            children.append(cp.deepcopy(pop[k]))

            # Calculate Levy flight for discrete, integer, and binary variables
            stepSize = np.round(step[k, :]*varID*(self.ub-self.lb))
            if all(stepSize == 0):
                stepSize = np.round(rand(len(varID))*(self.ub-self.lb))*varID
            children[i] = (children[i]+stepSize)%(self.ub+1-self.lb)

            #Build child applying variable boundaries
            children[i] = rejection_bounds(pop[k], children[i], stepSize,
                                           self.lb, self.ub)

        return children, used

    def cont_levy_flight(self, pop):
        """!
        Generate new children using Levy flights permutation of current
        generation design parameters according to:

        \f$ x_r^{g+1}=x_r^{g}+ \frac{1}{\beta} L_{\alpha,\gamma}, \f$

        where \f$ L_{\alpha,\gamma} \f$ is calculated in levy() according
        to the Mantegna algorithm.  Applies rejection_bounds() to ensure all
        solutions lie within the design space by adapting the step size to
        the size of the design space.

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        children = [] # Local copy of children generated
        varID = self.cID

        # Determine step size using Levy Flight
        step = levy(len(pop[0]), len(pop), alpha=self.alpha, gam=self.gamma,
                    n=self.n)

        # Perform global search from fracLevy*population parents
        used = []
        for i in range(int(self.fracLevy*self.population)):
            k = int(rand()*self.population)
            while k in used:
                k = int(rand()*self.population)
            used.append(k)
            children.append(cp.deepcopy(pop[k]))

            #Calculate Levy flight
            stepSize = 1.0/self.scalingFactor*step[k, :]*varID
            children[i] = children[i]+stepSize

            #Build child applying variable boundaries
            children[i] = rejection_bounds(pop[k], children[i], stepSize,
                                           self.lb, self.ub)

        return children, used

    def comb_levy_flight(self, pop):
        """!
        Generate new children using truncated Levy flights permutation and
        inversion of current generation design parameters.

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        # Initialize variables
        children = [] # Local copy of children generated
        used = []     # Parents randomly selected once already
        step = tlf(len(pop), len(pop[0]))

        for i in range(0, int(self.population*self.fracLevy)):

            # Randomly select parent
            k = int(rand()*self.population)
            while k in used:
                k = int(rand()*self.population)
            used.append(k)
            children.append(cp.deepcopy(pop[k]))

            # Create a tmp vector of only the combinatorial variables
            tmp = [children[-1][x] for x in range(0, len(self.xID)) \
                   if self.xID[x] == 1]

            # Invert the ordering based on levy flight step sizes
            for j in range(0, len(tmp)-1):
                flight = (tmp[j]+int(step[i][j]*len(tmp)))%(len(tmp)-1)
                if tmp[j+1] != flight:
                    ind = np.where(tmp == flight)[0][0]
                    if ind > j:
                        tmp[j+1:ind+1] = reversed(tmp[j+1:ind+1])
                    if j > ind:
                        tmp[ind:j+1] = reversed(tmp[ind:j+1])

            # Copy results back into child
            for x in range(0, len(self.xID)):
                if self.xID[x] == 1:
                    children[-1][x] = tmp[0]
                    del tmp[0]

        return children, used

    def scatter_search(self, pop):
        """!
        Generate new designs using the scatter search heuristic according to:

        \f$ x^{g+1} = c_1 + (c_2-c_1) r \f$

        where

        \f$ c_1 = x^e - d(1+\alpha \beta) \f$ \n
        \f$ c_2 = x^e - d(1-\alpha \beta) \f$ \n
        \f$ d = \frac{x^r - x^e}{2} \f$ \n \n

        and

        \f$ \alpha = \f$ 1 if i < j & -1 if i > j \n
        \f$ \beta = \frac{|j-i|-1}{b-2} \f$

        where b is the size of the population.

        Adapted from ideas in Egea, "An evolutionary method for complex-
        process optimization."

        Applies simple_bounds() to ensure all solutions lie within the design
        space by adapting the step size to the size of the design space.

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        # Update vectors
        intDiscID = self.iID+self.dID
        varID = self.cID

        # Use scatter search to generate candidate children solutions
        children = []
        used = []
        for i in range(0, int(len(pop)*self.fracElite), 1):
            #Randomly choose starting parent #2
            j = int(rand()*len(pop))
            while j == i or j in used:
                j = int(rand()*len(pop))
            used.append(i)

            d = (pop[j]-pop[i])/2.0
            if i < j:
                alpha = 1
            else:
                alpha = -1
            beta = (abs(j-i)-1)/(len(pop)-2)
            c1 = pop[i]-d*(1+alpha*beta)
            c2 = pop[i]+d*(1-alpha*beta)
            tmp = c1+(c2-c1)*rand(len(pop[i]))

            # Enforce integer and discrete constraints, if present
            tmp = tmp*varID+np.round(tmp*intDiscID)

            #Build child applying variable boundaries
            children.append(simple_bounds(tmp, self.lb, self.ub))

        return children, used

    def inversion_crossover(self, pop):
        """!
        Generate new designs by using inver-over on combinatorial variables.
        Adapted from ideas in Tao, "Iver-over Operator for the TSP."

        Although logic originally designed for combinatorial variables, it
        works for all variables and is used for all here.  The primary
        difference is the number of times that the crossover is performed.

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        """

        children, tmpNonComb, used = ([] for i in range(3))

        for i in range(0, int(len(pop)*self.fracElite), 1):
            # Randomly choose starting parent #2
            r = int(rand()*len(pop))
            while r == i:
                r = int(rand()*len(pop))

            # Break variables space into combinatorial and non-combinatorial
            if sum(self.cID+self.dID+self.iID) != 0:
                nonComb1 = pop[i][:np.where((self.cID+self.dID+self.iID) == 1)\
                              [0][-1]+1]
                nonComb2 = pop[r][:np.where((self.cID+self.dID+self.iID) == 1)\
                              [0][-1]+1]
            if sum(self.xID) != 0:
                comb1 = pop[i][:np.where(self.xID == 1)[0][-1]+1]
                comb2 = pop[r][:np.where(self.xID == 1)[0][-1]+1]

            # Randomly choose crossover point and break some ankles
            if sum(self.cID+self.dID+self.iID) != 0:
                c = int(rand()*len(nonComb1))
                if rand() > 0.5:
                    tmpNonComb.append(np.array(nonComb1[0:c+1].tolist() \
                                           +nonComb2[c+1:].tolist()))
                else:
                    tmpNonComb.append(np.array(nonComb2[0:c+1].tolist()\
                                           +nonComb1[c+1:].tolist()))
                # Track parents from whence ye came
                used.append(i)

            # Combinatorial crossover
            if sum(self.xID) != 0:
                # Loop over every item in the combinatorial set starting at a
                # randomly chosen pt
                c = int(rand()*len(comb1))
                for c1 in range(c, len(comb1), 1):
                    d2 = (contains_sublist(comb2, comb1[c1])+1)%len(comb1)
                    d1 = (contains_sublist(comb1, comb2[d2]))
                    c2 = (contains_sublist(comb2,
                                           comb1[(d1+1)%len(comb1)]))%len(comb1)

                    # Create first child
                    tmp1 = cp.copy(comb1)
                    if c1 < d1:
                        tmp1[c1+1:d1+1] = list(reversed(tmp1[c1+1:d1+1]))
                    else:
                        tmp1[d1:c1] = list(reversed(tmp1[d1:c1]))

                    # Create second child
                    tmp2 = cp.copy(comb2)
                    if c2 < d2:
                        tmp2[c2:d2] = list(reversed(tmp2[c2:d2]))
                    else:
                        tmp2[d2+1:c2+1] = list(reversed(tmp2[d2+1:c2+1]))

                    # Put the design vectors back together again into
                    # frankenchildren
                    if sum(self.cID+self.dID+self.iID) == 0 \
                          and sum(self.xID) != 0:
                        children.append(tmp1)
                        children.append(tmp2)
                    elif sum(self.cID+self.dID+self.iID) != 0 \
                          and sum(self.xID) != 0:
                        children.append(np.concatenate(tmpNonComb[-1],
                                                       tmp1))
                        children.append(np.concatenate(tmpNonComb[-1],
                                                       tmp2))

                    # Track parents from whence ye came
                    used.append(i)
                    used.append(r)

        # Put the design vectors back together again into frankenchildren
        if sum(self.cID+self.dID+self.iID) != 0 and sum(self.xID) == 0:
            children = tmpNonComb

        return children, used

    def crossover(self, pop):
        """!
        Generate new children using distance based crossover strategies on
        the top parent. Ideas adapted from Walton "Modified Cuckoo Search: A
        New Gradient Free Optimisation Algorithm" and Storn "Differential
        Evolution - A Simple and Efficient Heuristic for Global Optimization
        over Continuous Spaces"

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        # Update vectors
        intDiscID = self.iID+self.dID
        varID = self.cID

        # Initialize variables
        goldenRatio = (1.+sqrt(5))/2.
        dx = np.zeros_like(pop[0])

        # Crossover top parent with an elite parent to speed local convergence
        children = []
        used = []
        for i in range(0, int(self.fracElite*len(pop)), 1):
            r = int(rand()*self.population)
            while r in used or r == i:
                r = int(rand()*self.population)
            used.append(i)

            children.append(cp.deepcopy(pop[r]))
            dx = abs(pop[i]-children[i])/goldenRatio
            children[i] = children[i]+dx*varID+np.round(dx*intDiscID)
            children[i] = simple_bounds(children[i], self.lb, self.ub)

        return children, used

    def mutate(self, pop):
        """!
        Generate new children by adding a weighted difference between two
        population vectors to a third vector.  Ideas adapted from Storn,
        "Differential Evolution - A Simple and Efficient Heuristic for Global
        Optimization over Continuous Spaces" and Yang, "Nature Inspired
        Optimization Algorithms"

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        """

        # Update vectors
        intDiscID = self.iID+self.dID
        varID = self.cID

        children = []

        # Mutate; K is a status vector to see if mutation occurs
        k = rand(len(pop), len(pop[0])) > self.fracMutation*rand()

        #Bias the discovery to the worst fitness solutions
        childn1 = cp.copy(permutation(pop))
        childn2 = cp.copy(permutation(pop))

        #New solution by biased/selective random walks
        r = rand()
        for j in range(0, len(pop), 1):
            n = np.array((childn1[j]-childn2[j]))
            stepSize = r*n*varID+(n*intDiscID).astype(int)
            tmp = (pop[j]+stepSize*k[j, :])*varID+(pop[j]+stepSize*k[j, :]) \
                   *intDiscID%(self.ub+1-self.lb)
            children.append(simple_bounds(tmp, self.lb, self.ub))

        return children

    def two_opt(self, pop):
        """!
        Generate new children using the two_opt operator.

        Ideas adapted from:
        Lin and Kernighan, "An Effective Heurisic Algorithm for the Traveling
        Salesman Problem"

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        # Initialize variables
        children, used = ([] for i in range(2))

        for i in range(0, int(self.fracElite*len(pop)), 1):
            breaks = np.sort((rand(2)*len(pop[i])//1))
            breaks[1] = int(breaks[0] \
                            + tlf(1, 1)[0, 0]*len(pop[i]))%len(pop[i])
            while abs(breaks[0]-breaks[1]) < 2:
                breaks[1] = int(breaks[0] \
                            + tlf(1, 1)[0, 0]*len(pop[i]))%len(pop[i])
                np.sort(breaks)

            # Create the child
            children.append(pop[i])
            children[-1][int(breaks[0]):int(breaks[1])] = list(reversed(
                                  pop[i][int(breaks[0]):int(breaks[1])]))
            used.append(i)

        return children, used

    def three_opt(self, pop):
        """!
        Generate new children using the three_opt operator.

        Ideas adapted from:
        Lin and Kernighan, "An Effective Heurisic Algorithm for the Traveling
        Salesman Problem"

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        # Initialize variables
        children, used = ([] for i in range(2))

        for i in range(0, self.population, 1):
            tmp = []   # Make a local copy of current parent designs

            # Generate 3 random nodes
            breaks = np.sort(rand(3)*len(pop[i])//1)

            # Ensure that 3 different nodes are selected
            while breaks[1] == breaks[0] or breaks[1] == breaks[2]:
                breaks[1] = (rand()*len(pop[i])//1)
                breaks = np.sort(breaks)
            while breaks[2] == breaks[0]:
                breaks[2] = (rand()*len(pop[i])//1)
            breaks = np.sort(breaks)

            # Make reconnections first way
            tmp[0:int(breaks[0])] = pop[i][0:int(breaks[0])]
            tmp[len(tmp):int(len(tmp)+breaks[2]-breaks[1])] = \
                                       pop[i][int(breaks[1]):int(breaks[2])]
            tmp[len(tmp):int(len(tmp)+breaks[1]-breaks[0])] = \
                                       pop[i][int(breaks[0]):int(breaks[1])]
            tmp[len(tmp):int(len(tmp)+breaks[2]-len(pop[i]))] = \
                                       pop[i][int(breaks[2]):len(pop[i])]
            children.append(tmp)
            used.append(i)

            # Make reconnections second way
            tmp = []   # Reset local copy of current parent designs
            tmp[0:int(breaks[0])] = pop[i][0:int(breaks[0])]
            tmp[len(tmp):int(len(tmp)+breaks[1]-breaks[0])] = \
                        list(reversed(pop[i][int(breaks[0]):int(breaks[1])]))
            tmp[len(tmp):int(len(tmp)+breaks[2]-breaks[1])] = \
                        reversed(pop[i][int(breaks[1]):int(breaks[2])])
            tmp[len(tmp):int(len(tmp)+breaks[2]-len(pop[i]))] = \
                        pop[i][int(breaks[2]):len(pop[i])]
            children.append(tmp)
            used.append(i)

        return children, used

    def population_update(self, parents, children, timeline=None,
                          genUpdate=0, adoptedParents=[], mhFrac=0.0,
                          randomParents=False):
        """!
        Calculate fitness, apply constraints, if present, and update the
        population if the children are better than their parents. Several
        optional inputs are available to modify this process. Refer to the
        input param documentation for more details.

        @param parents: <em> list of parent objects </em> \n
            The current parents representing system designs. \n
        @param children: <em> list of arrays </em> \n
            The children design variables representing new system designs. \n
        @param timeline: <em> list of history objects </em> \n
            The histories of the optimization process containing best design,
            fitness, generation, and function evaluations. \n
        @param genUpdate: \e integer \n
            Indicator for how many generations to increment the counter by.
            Genenerally 0 or 1. \n
        @param adoptedParents: \e list \n
            A list of alternative parents to compare the children against.
            This alternative parents are then held accountable for not being
            better than the children of others. \n
        @param mhFrac: \e float \n
            The Metropolis-Hastings fraction.  A fraction of the otherwise
            discarded parents will be evaluated for acceptance against the
            greater population. \n
        @param randomParents: \e boolean \n
            If True, a random parent will be selected for comparison to the
            children. No one is safe. \n

        @return <em> list of parent objects: </em> The current parents
            representing system designs. \n
        @return \e integer:  The number of replacements made. \n
        @return <em> list of history objects: </em> If an initial timeline was
            provided, retunrs an updated history of the optimization process
            containing best design, fitness, generation, and function
            evaluations.
        """

        #Test input values for consistency
        assert hasattr(self.objective.func, '__call__'), ('Invalid \tion '
                                                        'handle.')

        if self.dID != []:
            assert np.sum(self.dID+self.xID) == len(self.discreteVals), ('A '
                       'map must exist for each discrete variable. {} '
                       'discrete variables, and {} maps provided.'.format(
                        np.sum(self.dID), len(self.discreteVals)))

        # Map the discrete variables for fitness calculation and storage
        if sum(self.dID)+sum(self.xID) != 0:
            for c in range(0, len(children)):
                children[c] = self.map_to_discretes(children[c])
            for p in parents:
                p.variables = self.map_to_discretes(p.variables)

        # Track # of replacements to track effectiveness of search methods
        replace = 0

        # Find worst fitness to use as the penalty
        for i in range(0, len(children), 1):
            fnew = self.objective.func(children[i])
            if fnew > self.penalty:
                self.penalty = fnew

        # Calculate fitness; replace parents if child has better fitness
        feval = 0
        for i in range(0, len(children), 1):
            if randomParents:
                j = int(rand()*len(parents))
            elif len(adoptedParents) == len(children):
                j = adoptedParents[i]
            else:
                j = i
            fnew = self.objective.func(children[i])
            for con in self.constraints:
                fnew += con.func(children[i])
            feval += 1
            if fnew < parents[j].fitness:
                parents[j].fitness = fnew
                parents[j].variables = cp.copy(children[i])
                parents[j].changeCount += 1
                parents[j].stallCount = 0
                replace += 1
                if parents[j].changeCount >= 25 and \
                      j >= self.population*self.fracElite:
                    parents[j].variables = self.initialize(1, 'random'
                                                          ).flatten()
                    parents[j].variables = self.map_to_discretes(
                                                  parents[j].variables)
                    fnew = self.objective.func(parents[j].variables)
                    for con in self.constraints:
                        fnew += con.func(parents[j].variables)
                    parents[j].fitness = fnew
                    parents[j].changeCount = 0
            else:
                parents[j].stallCount += 1
                if parents[j].stallCount > 50000 and j != 0:
                    parents[j].variables = self.initialize(1, 'random'
                                                          ).flatten()
                    parents[j].variables = self.map_to_discretes(
                                                  parents[j].variables)
                    fnew = self.objective.func(parents[j].variables)
                    for con in self.constraints:
                        fnew += con.func(parents[j].variables)
                    parents[j].fitness = fnew
                    parents[j].changeCount = 0
                    parents[j].stallCount = 0

                # Metropis-Hastings algorithm
                r = int(rand()*len(parents))
                if r <= mhFrac:
                    r = int(rand()*len(parents))
                    if fnew < parents[r].fitness:
                        parents[r].fitness = fnew
                        parents[r].variables = cp.copy(children[i])
                        parents[r].changeCount += 1
                        parents[r].stallCount += 1
                        replace += 1

        #Sort the pop
        parents.sort(key=lambda x: x.fitness)

        # If timeline provided, store history on timeline if new optimal
        # design found
        if timeline != None:
            if len(timeline) < 2:
                timeline.append(Event(1, feval, parents[0].fitness,
                                      parents[0].variables))
            elif parents[0].fitness < timeline[-1].fitness \
                  and abs((timeline[-1].fitness-parents[0].fitness) \
                          /parents[0].fitness) > self.convTol:
                timeline.append(Event(timeline[-1].generation,
                                      timeline[-1].evaluations+feval,
                                      parents[0].fitness, parents[0].variables))
            else:
                timeline[-1].generation += genUpdate
                timeline[-1].evaluations += feval

        # Map the discrete variables for fitness calculation and storage
        if sum(self.dID)+sum(self.xID) != 0:
            for p in parents:
                p.variables = self.map_from_discretes(p.variables)

        # Return updated population
        if timeline != None:
            return (parents, replace, timeline)
        else:
            return (parents, replace)

#------------------------------------------------------------------------------#
def simple_bounds(child, lb, ub):
    """!
    @ingroup GnoweeHeuristics
    Application of problem boundaries to generated solutions. If outside of the
    boundaries, the variable defaults to the boundary.

    @param child: \e array \n
        The proposed new system designs. \n
    @param lb: \e array \n
        The lower bounds of the design variable(s). \n
    @param ub: \e array \n
        The upper bounds of the design variable(s). \n

    @return \e array: The new system design that is within problem
        boundaries. \n
    """

    assert len(lb) == len(ub), ('Lower and upper bounds have different #s of '
                         'design variables in simple_bounds function.')
    assert len(lb) == len(child), ('Bounds and child have different #s of '
                         'design variables in simple_bounds function.')

    #Apply lower bound
    for i in range(0, len(child), 1):
        if child[i] < lb[i]:
            child[i] = lb[i]

    #Apply upper bound
    for i in range(0, len(child), 1):
        if child[i] > ub[i]:
            child[i] = ub[i]

    return child

#------------------------------------------------------------------------------#
def rejection_bounds(parent, child, stepSize, lb, ub):
    """!
    @ingroup GnoweeHeuristics
    Application of problem boundaries to generated solutions. Adjusts step size
    for all rejected solutions until within the boundaries.

    @param parent: \e array \n
        The current system designs. \n
    @param child: \e array \n
        The proposed new system designs. \n
    @param stepSize: \e float \n
        The stepsize for the permutation. \n
    @param lb: \e array \n
        The lower bounds of the design variable(s). \n
    @param ub: \e array \n
        The upper bounds of the design variable(s). \n

    @return \e array: The new system design that is within problem
        boundaries. \n
    """

    assert len(lb) == len(ub), ('Lower and upper bounds have different #s of '
                         'design variables in rejection_bounds function.')
    assert len(lb) == len(child), ('Bounds and child have different #s of '
                         'design variables in rejection_bounds function.')

    for i in range(0, len(child), 1):
        stepReductionCount = 0
        while child[i] < lb[i] or child[i] > ub[i]:
            if stepReductionCount >= 5:
                child[i] = cp.copy(parent[i])
            else:
                stepSize[i] = stepSize[i]/2.0
                child[i] = child[i]-stepSize[i]
                stepReductionCount += 1
    return child

#------------------------------------------------------------------------------#
def contains_sublist(lst, sublst):
    """!
    @ingroup GnoweeHeuristics
    Find index of sublist, if it exists.

    @param lst: \e list \n
        The list in which to search for sublst. \n
    @param sublst: \e list \n
        The list to search for. \n

    @return \e integer: Index location of sublst in lst. \n
    """
    for i in range(0, len(lst), 1):
        if sublst == lst[i]:
            return i
