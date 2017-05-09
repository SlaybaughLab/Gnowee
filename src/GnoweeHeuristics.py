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

@date 8May17
"""

import numpy as np
import copy as cp

from math import sqrt
from numpy.random import rand, permutation
from Sampling import levy, tlf, initial_samples

#------------------------------------------------------------------------------#
class GnoweeHeuristics(object):
    """!
    @ingroup GnoweeHeuristics
    The class is the foundation of the Gnowee optimization algorithm.  It sets
    the settings required for the algorithm and defines the heurstics.
    """

    ##
    def __init__(self, population=25, initSampling='lhc', fracDiscovered=0.2,
                 fracElite=0.2, fracLevy=0.2, alpha=1.5, gamma=1, n=1,
                 scalingFactor=10.0, penalty=0.0, maxGens=20000,
                 fevalMax=200000, convTol=1e-6, stallIterLimit=225,
                 optimalFitness=0, optConvTol=1e-2):
        """!
        Constructor to build the GnoweeHeuristics class.

        The default settings are found to be optimized for a wide range of
        problems, but can be changed to optimize performance for a particular
        problem type or class.  For more details, refer to the benchmark code
        in the development branch of the repo or <insert link to paper>.

        If the optimizal fitness is unknown, as it often is, this can be left
        as zero or some reasonable guess based on the understanding of the
        problem. If the opimtimal fitness is set below what is actually
        obatinable, the only impact is the removal of this convergence
        criteria, and the program will still run.

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param population: \e integer \n
            The number of members in each generation. \n
        @param initSampling: \e string \n
            The method used to sample the phase space and create the initial
            population. Valid options are 'random', 'nolh', 'nolh-rp',
            'nolh-cdr', and 'lhc' as specified in init_samples(). \n
        @param fracDiscovered : \e float \n
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
        @param fevalMax: \e integer \n
            The maximum number of objective function evaluations. \n
        @param convTol: \e float \n
            The minimum change of the best objective value before the search
            terminates. \n
        @param stallIterLimit: \e integer \n
            The maximum number of generations to search without a descrease
            exceeding convTol. \n
        @param optimalFitness: \e float \n
            The best know fitness value for the problem considered used to test
            for convergence. \n
        @param optConvTol: \e float \n
            The maximum deviation from the best know fitness value before the
            search terminates. \n
        """

        ## @var population
        # \e integer
        # The number of members in each generation.
        self.population = population

        ## @var initSampling
        # \e string
        # The method used to sample the phase space and create the initial
        # population. Valid options are 'random', 'nolh', 'nolh-rp',
        #'nolh-cdr', and 'lhc' as specified in init_samples().
        self.initSampling = initSampling

        ## @var fracDiscovered
        # \e float
        # Discovery probability used for the mutate() heuristic.
        self.fracDiscovered = fracDiscovered
        assert self.fracDiscovered >= 0 and self.fracDiscovered <= 1, 
                             'The probability of discovery must exist on (0,1]'
        ## @var fracElite
        # \e float
        # Elite fraction probability used for the scatter_search(), crossover(),
        # and cont_crossover() heuristics.
        self.fracElite = fracElite
        assert self.fracElite >= 0 and self.fracElite <= 1, ('The elitism ',
                                                'fraction must exist on (0,1]')

        ## @var fracLevy
        # \e float
        # Levy flight probability used for the disc_levy_flight() and
        # cont_levy_flight() heuristics.
        self.fracLevy = fracLevy
        assert self.fracLevy >= 0 and self.fracLevy <= 1, ('The probability ',
                        'that a Levy flight is performed must exist on (0,1]')

        ## @var alpha
        # \e float
        # Levy exponent - defines the index of the distribution and controls
        # scale properties of the stochastic process.
        self.alpha = alpha

        ## @var gamma
        # \e float
        # Gamma - scale unit of process for Levy flights.
        self.gamma = gamma

        ## @var n
        # \e integer
        # Number of independent variables - can be used to reduce Levy flight
        # sampling variance.
        self.n = n

        ## @var scalingFactor
        # \e float
        # Step size scaling factor used to adjust Levy flights to length scale
        # of system. The implementation of the Levy flight sampling makes this
        # largely arbitrary.
        self.scalingFactor = scalingFactor

        ## @var penalty
        # \e float
        # Individual constraint violation penalty to add to objective function.
        self.penalty = penalty

        ## @var maxGens
        # \e integer
        # The maximum number of generations to search.
        self.maxGens = maxGens

        ## @var fevalMax
        # \e integer
        # The maximum number of objective function evaluations.
        self.fevalMax = fevalMax

        ## @var convTol
        # \e float
        # The minimum change of the best objective value before the search
        # terminates.
        self.convTol = convTol

        ## @var stallIterLimit
        # \e integer
        # The maximum number of gen3rations to search without a descrease
        # exceeding convTol.
        self.stallIterLimit = stallIterLimit

        ## @var optimalFitness
        # \e float
        # The best know fitness value for the problem considered used to test
        # for convergence.
        self.optimalFitness = optimalFitness

        ## @var optConvTol
        # \e float
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
                                                            self.fracDiscovered,
                                                            self.fracElite,
                                                            self.fracLevy,
                                                            self.alpha,
                                                            self.gamma,
                                                            self.n,
                                                            self.scalingFactor,
                                                            self.penalty,
                                                            self.maxGens,
                                                            self.fevalMax,
                                                            self.convTol,
                                                            self.stallIterLimit,
                                                            self.optimalFitness,
                                                            self.optConvTol)

    def __str__(self):
        """!
        Human readable GnoweeHeuristics print function.

        @param self: <em> GnoweeHeuristics pointer </em> \n
            The GnoweeHeuristics pointer. \n
        """

        header = ["GnoweeHeuristics:"]
        header += ["Population = {}".format(self.population)]
        header += ["Sampling Method = {}".format(self.initSampling)]
        header += ["Discovery Fraction = {}".format(self.fracDiscovered)]
        header += ["Elitism Fraction = {}".format(self.fracElite)]
        header += ["Levy Fraction = {}".format(self.fracLevy)]
        header += ["Levy Alpha = {}".format(self.alpha)]
        header += ["Levy Gamma = {}".format(self.gamma)]
        header += ["Levy Independent Samples = {}".format(self.n)]
        header += ["Levy Scaling Parameter = {}".format(self.scalingFactor)]
        header += ["Constraint Violaition Penalty = {}".format(self.penalty)]
        header += ["Max # of Generations = {}".format(self.maxGens)]
        header += ["Max # of Function Evaluations = {}".format(self.fevalMax)]
        header += ["Convergence Tolerance = {}".format(self.convTol)]
        header += ["Stall Limit = {}".format(self.stallIterLimit)]
        header += ["Optimal Fitness = {}".format(self.optimalFitness)]
        header += ["Optimal Convergence Tolerance = {}".format(self.optConvTol)]
        return "\n".join(header)+"\n"

    def initialize(self, numSamples, sampleMethod, lb, ub, varType):
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
        @param lb: \e array \n
            The lower bounds of the design variable(s). \n
        @param ub: \e array \n
            The upper bounds of the design variable(s). \n
        @param varType: \e array \n
            The type of variable for each design parameter. Allowed values:
            'c' = continuous \n
            'i' = integer/binary (difference denoted by ub/lb) \n
            'd' = discrete where the allowed values are given by the option
                  discreteVals nxm arrary with n=# of discrete variables
                  and m=# of values that can be taken for each variable \n
            'x' = combinatorial. All of the variables denoted by x are
                  assumed to be "swappable" in combinatorial permutations.
                  There must be at least two variables denoted as
                  combinatorial. \n
            'f' = fixed design variable \n

        @return <em> list of arrays: </em> The initialized set of samples.
        """

        initSamples = initial_samples(lb, ub, sampleMethod, numSamples)
        for var in range(len(varType)):
            if varType[var] == 'i' or varType[var] == 'd':
                initSamples[:, var] = np.rint(initSamples[:, var])
        return initSamples

    def disc_levy_flight(self, pop, lb, ub, varID):
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
        @param lb: \e array \n
            The lower bounds of the design variable(s). \n
        @param ub: \e array \n
            The upper bounds of the design variable(s). \n
        @param varID: \e array \n
            A truth array indicating the location of the variables to be
            permuted. If the variable is to be permuted, a 1 is inserted at
            the variable location; otherwise a 0. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        assert len(lb) == len(ub), ('Lower and upper bounds have different ',
                       '#s of design variables in disc_levy_flight function.')
        assert len(lb) == len(pop[0]), ('Bounds and pop have different #s ',
                         'of design variables in disc_levy_flight function.')
        assert len(lb) == len(varID), ('The bounds size ({}) must be ',
             'consistent with the size of the variable ID truth vector ({}) ',
                'in disc_levy_flight function.').format(len(lb), len(varID))

        children = [] # Local copy of children generated

        # Determine step size using Levy Flight
        step = tlf(len(pop), len(pop[0]), alpha=self.alpha, gamma=self.gamma)

        # Initialize Variables
        used = []
        for i in range(0, int(self.fracLevy*self.population), 1):
            k = int(rand()*self.population)
            while k in used:
                k = int(rand()*self.population)
            used.append(k)
            children.append(cp.deepcopy(pop[k]))

            #Calculate Levy flight
            stepSize = np.round(step[k, :]*varID*(ub-lb))
            if all(stepSize == 0):
                stepSize = np.round(rand(len(varID))*(ub-lb))*varID
            children[i] = (children[i]+stepSize)%(ub+1-lb)

            #Build child applying variable boundaries
            children[i] = rejection_bounds(pop[k], children[i], stepSize, lb,
                                           ub)

        return children, used

    def cont_levy_flight(self, pop, lb, ub, varID):
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
        @param lb: \e array \n
            The lower bounds of the design variable(s). \n
        @param ub: \e array \n
            The upper bounds of the design variable(s). \n
        @param varID: \e array \n
            A truth array indicating the location of the variables to be
            permuted. If the variable is to be permuted, a 1 is inserted at
            the variable location; otherwise a 0. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        assert len(lb) == len(ub), ('Lower and upper bounds have different ',
                       '#s of design variables in cont_levy_flight function.')
        assert len(lb) == len(pop[0]), ('Bounds and pop have different #s ',
                         'of design variables in cont_levy_flight function.')
        assert len(lb) == len(varID), ('The bounds size ({}) must be ',
             'consistent with the size of the variable ID truth vector ({}) ',
                'in cont_levy_flight function.').format(len(lb), len(varID))

        children = [] # Local copy of children generated

        # Determine step size using Levy Flight
        step = levy(len(pop[0]), len(pop), alpha=self.alpha, gamma=self.gamma,
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
            children[i] = rejection_bounds(pop[k], children[i], stepSize, lb,
                                           ub)

        return children, used

    def scatter_search(self, pop, lb, ub, varID, intDiscID=None):
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
        @param lb: \e array \n
            The lower bounds of the design variable(s). \n
        @param ub: \e array \n
            The upper bounds of the design variable(s). \n
        @param varID: \e array \n
            A truth array indicating the location of the variables to be
            permuted. If the variable is to be permuted, a 1 is inserted at
            the variable location; otherwise a 0. \n
        @param intDiscID: \e array \n
            A truth array indicating the location of the discrete variable.
            A 1 is inserted at the discrete variable location; otherwise a
            0. If no discrete variables, the array will be set to 0
            automatically. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        assert len(lb) == len(ub), ('Lower and upper bounds have different ',
                       '#s of design variables in scatter_search function.')
        assert len(lb) == len(pop[0]), ('Bounds and pop have different #s ',
                         'of design variables in scatter_search function.')
        assert len(lb) == len(varID), ('The bounds size ({}) must be ',
             'consistent with the size of the variable ID truth vector ({}) ',
                'in scatter_search function.').format(len(lb), len(varID))

        # If no discretes variables exist, set ID array to zero
        if len(intDiscID) != len(varID):
            intDiscID = np.zeros_like(varID)

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
            children.append(simple_bounds(tmp, lb, ub))

        return children, used

    def crossover(self, pop):
        """!
        Generate new designs by using inver-over on combinatorial variables.
        Adapted from ideas in Tao, "Iver-over Operator for the TSP."

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        """

        children = []

        for i in range(0, int(len(pop)*self.fracElite), 1):
            #Randomly choose starting parent #2
            r = int(rand()*len(pop))
            while r == i:
                r = int(rand()*len(pop))

            # Randomly choose crossover point
            c = int(rand()*len(pop[i]))
            if rand() < 0.5:
                children.append(np.array(pop[i][0:c+1].tolist() \
                                         +pop[r][c+1:].tolist()))
            else:
                children.append(np.array(pop[r][0:c+1].tolist()\
                                         +pop[i][c+1:].tolist()))

        return children

    def cont_crossover(self, pop, lb, ub, varID, intDiscID=None):
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
        @param lb: \e array \n
            The lower bounds of the design variable(s). \n
        @param ub: \e array \n
            The upper bounds of the design variable(s). \n
        @param varID: \e array \n
            A truth array indicating the location of the variables to be
            permuted. If the variable is to be permuted, a 1 is inserted at
            the variable location; otherwise a 0. \n
        @param intDiscID: \e array \n
            A truth array indicating the location of the discrete variable.
            A 1 is inserted at the discrete variable location; otherwise a
            0. If no discrete variables, the array will be set to 0
            automatically. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """

        assert len(lb) == len(ub), ('Lower and upper bounds have different ',
                       '#s of design variables in cont_crossover function.')
        assert len(lb) == len(pop[0]), ('Bounds and pop have different #s ',
                         'of design variables in cont_crossover function.')
        assert len(lb) == len(varID), ('The bounds size ({}) must be ',
             'consistent with the size of the variable ID truth vector ({}) ',
                'in cont_crossover function.').format(len(lb), len(varID))

        # If no discretes variables exist, set ID array to zero
        if len(intDiscID) != len(varID):
            intDiscID = np.zeros_like(varID)

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
            children[i] = simple_bounds(children[i], lb, ub)

        return children, used

    def mutate(self, pop, lb, ub, varID, intDiscID=None):
        """!
        Generate new children by adding a weighted difference between two
        population vectors to a third vector.  Ideas adapted from Storn,
        "Differential Evolution - A Simple and Efficient Heuristic for Global
        Optimization over Continuous Spaces" and Yang, "Nature Inspired
        Optimmization Algorithms"

        @param self: <em> GnoweeHeuristic pointer </em> \n
            The GnoweeHeuristics pointer. \n
        @param pop: <em> list of arrays </em> \n
            The current parent sets of design variables representing system
            designs for the population. \n
        @param lb: \e array \n
            The lower bounds of the design variable(s). \n
        @param ub: \e array \n
            The upper bounds of the design variable(s). \n
        @param varID: \e array \n
            A truth array indicating the location of the variables to be
            permuted. If the variable is to be permuted, a 1 is inserted at
            the variable location; otherwise a 0. \n
        @param intDiscID: \e array \n
            A truth array indicating the location of the discrete variable.
            A 1 is inserted at the discrete variable location; otherwise a
            0. If no discrete variables, the array will be set to 0
            automatically. \n

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        """

        assert len(lb) == len(ub), ('Lower and upper bounds have different ',
                       '#s of design variables in mutate function.')
        assert len(lb) == len(pop[0]), ('Bounds and pop have different #s ',
                         'of design variables in mutate function.')
        assert len(lb) == len(varID), ('The bounds size ({}) must be ',
             'consistent with the size of the variable ID truth vector ({}) ',
                'in mutate function.').format(len(lb), len(varID))

        children = []

        # If no discretes variables exist, set ID array to zero
        if len(intDiscID) != len(varID):
            intDiscID = np.zeros_like(varID)

        #Discover (1-fd); K is a status vector to see if discovered
        k = rand(len(pop), len(pop[0])) > self.fracDiscovered

        #Bias the discovery to the worst fitness solutions
        childn1 = cp.copy(permutation(pop))
        childn2 = cp.copy(permutation(pop))

        #New solution by biased/selective random walks
        r = rand()
        for j in range(0, len(pop), 1):
            n = np.array((childn1[j]-childn2[j]))
            stepSize = r*n*varID+(n*intDiscID).astype(int)
            tmp = (pop[j]+stepSize*k[j, :])*varID+(pop[j]+stepSize*k[j, :]) \
                   *intDiscID%(ub+1-lb)
            children.append(simple_bounds(tmp, lb, ub))

        return children

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

    assert len(lb) == len(ub), ('Lower and upper bounds have different #s of ',
                         'design variables in simple_bounds function.')
    assert len(lb) == len(child), ('Bounds and child have different #s of ',
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

    assert len(lb) == len(ub), ('Lower and upper bounds have different #s of ',
                         'design variables in rejection_bounds function.')
    assert len(lb) == len(child), ('Bounds and child have different #s of ',
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
