"""!
@file src/GnoweeUtilities.py
@package Gnowee

@defgroup GnoweeUtilities GnoweeUtilities

@brief Classes and methods to support the Gnowee optimization algorithm.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""

import numpy as np
import copy as cp

from Constraints import Constraint
from ObjectiveFunction import ObjectiveFunction

#------------------------------------------------------------------------------#
class Parent(object):
    """!
    @ingroup GnoweeUtilities
    The class contains all of the parameters pertinent to a member of the
    population.
    """

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

        ## @var variables
        # \e array:
        # The set of variables representing a design solution.
        self.variables = variables

        ## @var fitness
        # \e float:
        # The assessed fitness for the current set of variables.
        self.fitness = fitness

        ## @var changeCount
        # \e integer:
        # The number of improvements to the current population member.
        self.changeCount = changeCount

        ## @var stallCount
        # \e integer:
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

        ## @var generation
        # \e integer:
        # The generation the design was arrived at.
        self.generation = generation

        ## @var evaluations
        # \e integer:
        # The number of fitness evaluations done to obtain this design.
        self.evaluations = evaluations

        ## @var fitness
        # \e float:
        # The assessed fitness for the current set of variables.
        self.fitness = fitness

        ## @var design
        # \e array:
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

#------------------------------------------------------------------------------#
class ProblemParameters(object):
    """!
    @ingroup GnoweeUtilities
    Creates an object containing key features of the chosen optimization
    problem. The methods provide a way of predefining problems for repeated use.
    """

    def __init__(self, objective=None, constraints=[], lowerBounds=[],
                 upperBounds=[], varType=[], discreteVals=[], optimum=0.0,
                 pltTitle='', histTitle='', varNames=['']):
        """!
        Constructor for the ProblemParameters class. The default constructor
        is useless for an optimization, but allows a placeholder class to be
        instantiated.

        This class contains the problem definitions required for an
        optimization problem. It allows for single objective, multi-constraint
        mixed variable optimization and any subset thereof. At a minimum,
        the objective, lowerBounds, upperBounds, and varType attributes must be
        specified to run Gnowee.

        The optimum is used for convergence criteria and can be input if
        known. If not, the default (0.0) will suffice for most problems,
        or the user can make an educated guess based on their knowledge of
        the problem.

        @param self: <em> ProblemParameters pointer </em> \n
            The ProblemParameters pointer. \n
        @param objective: <em> ObjectiveFunction object </em> \n
            The optimization objective function to be used.  Only a single
            objective function can be specified. \n
        @param constraints: <em> list of Constraint objects </em> \n
            The constraints on the problem. Zero constraints can be specified
            as an empty list ([]), or multiple constraints can be specified
            as a list of Constraint objects. \n
        @param lowerBounds: \e array \n
            The lower bounds of the design variable(s). Only enter the bounds
            for continuous and integer/binary variables. The order must match
            the order specified in varType and ub. \n
        @param upperBounds: \e array \n
            The upper bounds of the design variable(s). Only enter the bounds
            for continuous and integer/binary variables. The order must match
            the order specified in varType and lb. \n
        @param varType: <em> list or array </em> \n
            The type of variable for each position in the upper and lower
            bounds array. Discrete and combinatorial variables are to be
            included last as they are specified separately from the lb/ub
            through the discreteVals optional input. The order should be
            the same as shown below. \n
            Allowed values: \n
             'c' = continuous over a given range (range specified in lb &
                   ub). \n
             'i' = integer/binary (difference denoted by ub/lb). \n
             'f' = fixed design variable. Will not be considered of any
                   permutation. \n
             'd' = discrete where the allowed values are given by the option
                   discreteVals nxm arrary with n=# of discrete variables and
                   m=# of values that can be taken for each variable. \n
             'x' = combinatorial. All of the variables denoted by x are assumed
                   to be "swappable" in combinatorial permutations and assumed
                   to take discrete values specified in by discreteVals. There
                   must be at least two variables denoted as combinatorial.
                   The algorithms are only set up to handle one set of
                   combinatorial variables per optimization problem.
                   Combinatorial variales should be specified last and as a
                   contiguous group. \n
        @param discreteVals: <em> list of list(s) </em> \n
            nxm with n=# of discrete and combinatorial variables and m=# o
            f values that can be taken for each variable. For example, if you
            had two variables representing the tickness and diameter of a
            cylinder that take standard values, the discreteVals could be
            specified as: \n
            discreteVals = [[0.125, 0.25, 0.375], [0.25, 0.5, 075]] \n
            For combinatorial problems, you must specify the same possible
            values that can be taken n times, where n is the number of different
            positions in the combinatorial sequence. suppose you had a gear that
            could be placed at position 2, 3, 4, or 5. The discreteVals would be
            specified as (assuming no other discretes): \n
            discreteVals = [[2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5],
            [2, 3, 4, 5]] \ n
            Gnowee will then map the optimization results to these allowed
            values. \n
        @param optimum: \e float \n
            The global optimal solution. \n
        @param pltTitle: \e string \n
            The title used for plotting the results of the optimization. \n
        @param histTitle: \e string \n
            The plot title for the histogram of the optimization results. \n
        @param varNames: <em> list of strings </em>
            The names of the variables for the optimization problem. \n
        """

        ## @var objective
        # <em> ObjectiveFunction Object: </em> The objective function object
        # to be used for the optimization.
        self.objective = objective

        ## @var constraints
        # <em> list of Constraint Objects: </em> The constraints on the
        # optimization design space.
        if type(constraints) != list:
            self.constraints = [constraints]
        else:
            self.constraints = constraints

        ## @var lb
        # \e array: The lower bounds of the design variable(s).
        self.lb = lowerBounds

        ## @var ub
        # \e array: The upper bounds of the design variable(s).
        self.ub = upperBounds

        ## @var varType
        # \e array: The type of variable for each position in the upper and
        # lower bounds array.
        self.varType = varType

        ## @var discreteVals
        #\e array: nxm with n=# of discrete variables and m=# of values that
        # can be taken for each variable.
        self.discreteVals = discreteVals

        ## @var optimum
        # \e float: The global optimal solution.
        self.optimum = optimum

        ## @var pltTitle
        # \e string: The title used for plotting the results of the
        # optimization.
        self.pltTitle = pltTitle

        ## @var histTitle
        # \e string: The plot title for the histogram of the optimization
        # results.
        self.histTitle = histTitle

        ## @var varNames
        # <em> list of strings: </em> The names of the variables for the
        # optimization problem.
        self.varNames = varNames

        # Ensure that the correct inputs were provided; modify as neccesary
        # to meet Gnowee's requirements;
        # Populate variable type id vectors
        if len(self.lb) and len(self.ub) and len(self.varType) != 0 \
                        or len(self.discreteVals) and len(varType) != 0:
            self.sanitize_inputs()

            ## @var cID:
            # \e array: The continuous variable truth array. This contains
            # a one in the positions corresponding to continuous variables
            # and 0 otherwise.
            self.cID = []

            ## @var iID:
            # \e array: The integer variable truth array. This contains
            # a one in the positions corresponding to continuous variables
            # and 0 otherwise.
            self.iID = []

            ## @var dID:
            # \e array: The discrete variable truth array. This contains
            # a one in the positions corresponding to continuous variables
            # and 0 otherwise.
            self.dID = []

            ## @var xID:
            # \e array: The combinatorial variable truth array. This contains
            # a one in the positions corresponding to continuous variables
            # and 0 otherwise.
            self.xID = []

            # Develop ID vectors for each variable type
            for var in range(len(self.varType)):
                if 'c' in self.varType[var]:
                    self.cID.append(1)
                else:
                    self.cID.append(0)
                if 'i' in self.varType[var]:
                    self.iID.append(1)
                else:
                    self.iID.append(0)
                if 'd' in self.varType[var]:
                    self.dID.append(1)
                else:
                    self.dID.append(0)
                if 'x' in self.varType[var]:
                    self.xID.append(1)
                else:
                    self.xID.append(0)
            self.cID = np.array(self.cID)
            self.iID = np.array(self.iID)
            self.dID = np.array(self.dID)
            self.xID = np.array(self.xID)

    def __repr__(self):
        """!
        ProblemParameters class attribute print function.

        @param self: <em> pointer </em> \n
            The ProblemParameters pointer. \n
        """
        return ("ProblemParameters({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "
                "{}, {}, {})".format(self.objective, self.constraints, self.lb,
                             self.ub, self.varType, self.discreteVals,
                             self.optimum, self.pltTitle, self.histTitle,
                             self.varNames, self.cID, self.iID, self.dID,
                             self.xID))

    def __str__(self):
        """!
        Human readable ProblemParameters print function.

        @param self: \e pointer \n
            The ProblemParameters pointer. \n
        """

        header = ["  ProblemParameters:"]
        header += ["Lower Bounds: {}".format(self.lb)]
        header += ["Upper Bounds: {}".format(self.ub)]
        header += ["Variable Types: {}".format(self.varType)]
        header += ["Continuous ID Vector: {}".format(self.cID)]
        header += ["Integer ID Vector: {}".format(self.iID)]
        header += ["Discrete ID Vector: {}".format(self.dID)]
        header += ["Combinatorial ID Vector: {}".format(self.xID)]
        if len(self.discreteVals) > 1:
            if self.discreteVals[0] == self.discreteVals[1]:
                header += [("Discrete Values (only printing elem 1 of {}): "
                           "{}".format(len(self.discreteVals[0]),
                                     self.discreteVals[0]))]
        else:
            header += [("Discrete Values: {} ".format(self.discreteVals))]
        header += ["Global Optimum: {}".format(self.optimum)]
        header += ["Plot Title: {}".format(self.pltTitle)]
        header += ["Histogram Title: {}".format(self.histTitle)]
        header += ["Variable Names: {}".format(self.varNames)]
        header += ["{}".format(self.objective)]
        for con in self.constraints:
            header += ["{}".format(con)]
        return "\n".join(header)+"\n"

    def sanitize_inputs(self):
        """!
        Checks and cleans user inputs to be compatible with expectations from
        the Gnowee algorithm.

        @param self: \e pointer \n
            The ProblemParameters pointer. \n
        """

        # Check input variables
        assert self.varType.count('d') + self.varType.count('x') \
                             == len(self.discreteVals), ('The '
                            'allowed discrete  values must be specified for '
                            'each discrete variable. {} in varType, but {} in '
                            'discreteVals.'.format(self.varType.count('d')+\
                            self.varType.count('x'), len(self.discreteVals)))
        assert self.varType.count('c')+self.varType.count('i') == len(self.ub)\
                            , ('Each specified continuous, binary, and '
                            'integer variable must  have a corresponding '
                            'upper and lower bounds. {}  variables and {} '
                            'bounds specified'.format(self.varType.count('c')\
                             + self.varType.count('i'), len(self.lb)))
        assert max(len(self.varType) - 1 - self.varType[::-1].index('c') \
                     if 'c' in self.varType else -1,
                   len(self.varType) - 1 - self.varType[::-1].index('i') \
                     if 'i' in self.varType else -1) \
                    < self.varType.index('d') if 'd' in self.varType else \
                      len(self.varType), ('The discrete variables must be '
                    'specified after the continuous, binary, and integer '
                    'variables. The order given was {}'.format(self.varType))
        assert len(self.lb) == len(self.ub), ('The lower and upper bounds must '
                    'have the same dimensions. lb = {}, ub = {}'.format(
                    len(self.lb), len(self.ub)))
        assert set(self.varType).issubset(['c', 'i', 'd', 'x', 'f']), ('The '
                    'variable specifications do not match the allowed values '
                    'of "c", "i", "d", "x", "f". The varTypes specified is  '
                    '{}'.format(self.varType))
        if len(self.ub) != 0 and len(self.lb) != 0:
            assert np.all(self.ub > self.lb), ('All upper-bound values must '
                    'be greater than lower-bound values')

        # Ensure discreteVals is a list
        if type(self.discreteVals) != list:
            self.discreteVals = self.discreteVals.tolist()

        #  Append discretes to lb and ubs and convert to numpy arrays
        for d in range(len(self.discreteVals)):
            self.lb.append(0)
            self.ub.append(len(self.discreteVals[d])-1)

        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)

    def map_to_discretes(self, variables):
        """!
        Maps the sampled discrete indices to the array of allowable discrete
        values and returns the associated variable array.

        @param self: \e pointer \n
            The ProblemParameters pointer. \n
            The Parent pointer. \n
        @param variables: \e array \n
            The set of variables representing a design solution. \n

        @return \e array: An array containing the variables associated with
            the design.
        """

        varID = self.dID+self.xID
        if sum(varID) != 0:
            tmpVar = []
            i = 0
            for j in range(len(varID)):
                if varID[j] == 1:
                    tmpVar.append(self.discreteVals[i][int(variables[j])])
                    i += 1
                elif self.iID[j] == 1:
                    tmpVar.append(int(variables[j]))
                else:
                    tmpVar.append(variables[j])
        else:
            tmpVar = cp.copy(variables)
        return np.array(tmpVar)

    def map_from_discretes(self, variables):
        """!
        Maps the discrete values to indices for sampling.

        @param self: \e pointer \n
            The ProblemParameters pointer. \n
            The Parent pointer. \n
        @param variables: \e array \n
            The set of variables representing a design solution. \n

        @return \e array: An array containing the variables associated with
            the design.
        """

        varID = self.dID+self.xID
        variables = variables.tolist()
        if sum(varID) != 0:
            tmpVar = []
            i = 0
            for j in range(len(varID)):
                if varID[j] == 1:
                    tmpVar.append(self.discreteVals[i].index(variables[j]))
                    i += 1
                else:
                    tmpVar.append(variables[j])
        else:
            tmpVar = cp.copy(variables)
        return np.array(tmpVar)

    def set_preset_params(self, funct, algorithm='Gnowee', dimension=2):
        """!
        Instantiates a ProblemParameters object and populations member
        variables from a set of predefined problem types.

        @param self: \e pointer \n
            The ProblemParameters pointer. \n
        @param funct: \e string \n
            Name of function being optimized. \n
        @param algorithm: \e string \n
            Name of optimization program used. \n
        @param dimension: \e integer \n
            Used to set the dimension for scalable problems. \n
        """

        # Build temp varType array for continuous problems with variable
        # dimensions
        v = []
        for i in range(0, dimension):
            v.append('c')

        for case in Switch(funct):
            if case('mi_spring'):
                ProblemParameters.__init__(self, ObjectiveFunction('mi_spring'),
                                 Constraint('mi_spring', 0.0), [0.01, 1],
                                 [3.0, 10], ['c', 'i', 'd'],
                                 [[0.009, 0.0095, 0.0104, 0.0118,
                                   0.0128, 0.0132, 0.014, 0.015, 0.0162,
                                   0.0173, 0.018, 0.020, 0.023, 0.025, 0.028,
                                   0.032, 0.035, 0.041, 0.047, 0.054, 0.063,
                                   0.072, 0.080, 0.092, 0.105, 0.120, 0.135,
                                   0.148, 0.162, 0.177, 0.192, 0.207, 0.225,
                                   0.244, 0.263, 0.283, 0.307, 0.331, 0.362,
                                   0.394, 0.4375, 0.500]], 2.65856,
                                 ('\\textbf{MI Spring Optimization using %s}'
                                  %algorithm),
                                 ('\\textbf{Function Evaluations for Spring '
                                  'Optimization using %s}' %algorithm),
                                 ['\\textbf{Fitness}', '\\textbf{Spring Diam}',
                                  '\\textbf{\# Coils}', '\\textbf{Wire Diam}'])
                break
            if case('spring'):
                ProblemParameters.__init__(self, ObjectiveFunction('spring'),
                                  Constraint('spring', 0.0), [0.05, 0.25, 2.0],
                                  [2.0, 1.3, 15.0], ['c', 'c', 'c'], [],
                                  0.012665,
                                  ('\\textbf{Spring Optimization using %s}'
                                   %algorithm),
                                  ('\\textbf{Function Evaluations for Spring '
                                   'Optimization using %s}' %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{Width}',
                                  '\\textbf{Diameter}', '\\textbf{Length}'])
                break
            if case('pressure_vessel'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('pressure_vessel'),
                                  Constraint('pressure_vessel', 0.0),
                                  [0.0625, 0.0625, 10.0, 1E-8],
                                  [1.25, 99*0.0625, 50.0, 200.0],
                                  ['c', 'c', 'c', 'c'], [], 5885.332800,
                                  ('\\textbf{Pressure Vessel Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for '
                                   'Pressure Vessel Optimization using %s}'
                                    %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{Thickness}',
                                   '\\textbf{Head Thickness}',
                                   '\\textbf{Inner Radius}',
                                   '\\textbf{Cylinder Length}'])
                break
            if case('mi_pressure_vessel'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('mi_pressure_vessel'),
                                  Constraint('mi_pressure_vessel', 0.0),
                                  [10.0, 1E-8], [50.0, 200.0],
                                  ['c', 'c', 'd', 'd'],
                                  [(np.asarray(range(99))*0.0625+0.0625)\
                                                                     .tolist(),
                                   (np.asarray(range(99))*0.0625+0.0625)\
                                                                     .tolist()],
                                  6059.714335,
                                  ('\\textbf{MI Pressure Vessel Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for '
                                   'MI Pressure Vessel Optimization using %s}'
                                    %algorithm),
                                  ['\\textbf{Fitness}',
                                   '\\textbf{Inner Radius}',
                                   '\\textbf{Cylinder Length}',
                                   '\\textbf{Shell Thickness}',
                                   '\\textbf{Head Thickness}'])
                break
            if case('welded_beam'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('welded_beam'),
                                  Constraint('welded_beam', 0.0),
                                  [0.1, 0.1, 1E-8, 1E-8],
                                  [10.0, 10.0, 10.0, 2.0],
                                  ['c', 'c', 'c', 'c'], [], 1.724852,
                                  ('\\textbf{Welded Beam Optimization using '
                                   '%s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Welded '
                                   'Beam Optimization using %s}' %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{Weld H}',
                                   '\\textbf{Weld L}', '\\textbf{Beam H}',
                                   '\\textbf{Beam W}'])
                break
            if case('speed_reducer'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('speed_reducer'),
                                  Constraint('speed_reducer', 0.0),
                                  [2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0],
                                  [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5],
                                  ['c', 'c', 'c', 'c', 'c', 'c', 'c'], [],
                                  2996.348165,
                                  ('\\textbf{Speed Reducer Optimization using '
                                   '%s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Speed '
                                   'Reducer Optimization using %s}' %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{Face Width}',
                                   '\\textbf{Module}', '\\textbf{Pinion Teeth}',
                                   '\\textbf{1st Shaft L}',
                                   '\\textbf{2nd Shaft L}',
                                   '\\textbf{1st Shaft D}',
                                   '\\textbf{2nd Shaft D}'])
                break
            if case('mi_chemical_process'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('mi_chemical_process'),
                                  Constraint('mi_chemical_process', 0.0),
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [10.0, 10.0, 10.0, 1, 1, 1, 1],
                                  ['c', 'c', 'c', 'i', 'i', 'i', 'i'], [],
                                  4.579582,
                                  ('\\textbf{MI Chemical Process Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Chemical '
                                   'Process Optimization using %s}' %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{x1}',
                                   '\\textbf{x2}', '\\textbf{x3}',
                                   '\\textbf{y1}', '\\textbf{y2}',
                                   '\\textbf{y3}', '\\textbf{y4}'])
                break
            if case('dejong'):
                ProblemParameters.__init__(self, ObjectiveFunction('dejong'),
                                  [], np.ones(dimension)*-5.12,
                                  np.ones(dimension)*5.12, v, [], 0.0000,
                                  ('\\textbf{De Jong Function Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for De Jong '
                                   'Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)])
                break
            if case('shifted_dejong'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('shifted_dejong'),
                                  [], np.ones(dimension)*-5.12,
                                  np.ones(dimension)*-5.12, v, [], 0.0000,
                                  ('\\textbf{Shifted De Jong Function '
                                   'Optimization using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Shifted '
                                   'De Jong Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)])
                break
            if case('ackley'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('ackley'),
                                  [], np.ones(dimension)*-25.,
                                  np.ones(dimension)*25., v, [], 0.0000,
                                  ('\\textbf{Ackley Function Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Ackley '
                                   'Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                    %i for i in range(dimension)])
                break
            if case('shifted_ackley'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('shifted_ackley'),
                                  [], np.ones(dimension)*-25.,
                                  np.ones(dimension)*25., v, [], 0.0000,
                                  ('\\textbf{Shifted Ackley Function '
                                   'Optimization using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Shifted '
                                   'Ackley Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)])
                break
            if case('easom'):
                ProblemParameters.__init__(self, ObjectiveFunction('easom'),
                                  [], np.array([-100., -100.]),
                                  np.array([100., 100.]), v, [], -1.0000,
                                  ('\\textbf{Easom Function Optimization using '
                                   '%s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Easom '
                                   'Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{x}',
                                   '\\textbf{y}'])
                break
            if case('shifted_easom'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('shifted_easom'),
                                  [], np.array([-100., -100.]),
                                  np.array([100., 100.]), v, [], -1.0000,
                                  ('\\textbf{Shifted Easom Function '
                                   'Optimization using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Shifted '
                                   'Easom Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{x}',
                                   '\\textbf{y}'])
                break
            if case('griewank'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('griewank'),
                                  [], np.ones(dimension)*-600.,
                                  np.ones(dimension)*600., v, [], 0.0000,
                                  ('\\textbf{Griewank Function Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Griewank '
                                   'Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)])
                break
            if case('shifted_griewank'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('shifted_griewank'),
                                  [], np.ones(dimension)*-600.,
                                  np.ones(dimension)*600., v, [], 0.0000,
                                  ('\\textbf{Shifted Easom Function '
                                   'Optimization using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Shifted '
                                   'Easom Function Optimization using %s}'
                                   %algorithm,
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)]))
                break
            if case('rastrigin'):
                ProblemParameters.__init__(self, ObjectiveFunction('rastrigin'),
                                  [], np.ones(dimension)*-5.12,
                                  np.ones(dimension)*5.12, v, [], 0.0000,
                                  ('\\textbf{Rastrigin Function Optimization '
                                  'using %s}' %algorithm), \
                                  ('\\textbf{Function Evaluations for '
                                   'Rastrigin Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)])
                break
            if case('shifted_rastrigin'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('shifted_rastrigin'),
                                  [], np.ones(dimension)*-5.12,
                                  np.ones(dimension)*5.12, v, [], 0.0000,
                                  ('\\textbf{Shifted Rastrigin Function '
                                   'Optimization using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Shifted '
                                   'Rastrigin Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)])
                break
            if case('rosenbrock'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('rosenbrock'),
                                  [], np.ones(dimension)*-5.,
                                  np.ones(dimension)*5., v, [], 0.0000,
                                  ('\\textbf{Rosenbrock Function Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for '
                                   'Rosenbrock Function Optimization using '
                                   '%s}' %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                    %i for i in range(dimension)])
                break
            if case('shifted_rosenbrock'):
                ProblemParameters.__init__(self,
                                  ObjectiveFunction('shifted_rosenbrock'),
                                  [], np.ones(dimension)*-5.,
                                  np.ones(dimension)*5., v, [], 0.0000,
                                  ('\\textbf{Shifted Rosenbrock Function '
                                   'Optimization using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Shifted '
                                   'Rosenbrock Function Optimization using %s}'\
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)])
                break
            if case('tsp'):
                ProblemParameters.__init__(self, ObjectiveFunction('tsp'),
                                  [], [], [], [], [], 0.0,
                                  ('\\textbf{TSP Optimization using %s}'
                                   %algorithm),
                                  ('\\textbf{Function Evaluations for TSP '
                                   'using %s}' %algorithm),
                                  ['\\textbf{Fitness}'])
                break
            if case():
                print ('ERROR: Fishing in the deep end you are. Define your '
                       'own parameter set you must.')

#------------------------------------------------------------------------------#
class Switch(object):
    """!
    @ingroup GnoweeUtilities
    Creates a switch class object to switch between cases.
    """

    def __init__(self, value):
        """!
        Case constructor.

        @param self: <em> pointer </em> \n
            The Switch pointer. \n
        @param value: \e string \n
            Case selector value. \n
        """

        ## @var value
        # \e string: Case selector value.
        self.value = value

        ## @var fall
        # \e boolean: Match indicator.
        self.fall = False

    def __iter__(self):
        """!
        Return the match method once, then stop.

        @param self: <em> pointer </em> \n
            The Switch pointer. \n
        """
        yield self.match
        raise StopIteration

    def match(self, *args):
        """!
        Indicate whether or not to enter a case suite.

        @param self: <em> pointer </em> \n
            The Switch pointer. \n
        @param *args: \e list \n
            List of comparisons. \n

        @return \e boolean: Outcome of comparison match
        """
        if self.fall or not args:
            return True
        elif self.value in args:
            self.fall = True
            return True
        else:
            return False
