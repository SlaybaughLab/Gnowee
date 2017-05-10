"""!
@file src/GnoweeUtilities.py
@package Gnowee

@defgroup GnoweeUtilities GnoweeUtilities

@brief Classes and methods to support the Gnowee optimization algorithm.

@author James Bevins

@date 9May17
"""

import numpy as np
import copy as cp
import bisect

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

    def __init__(self, lowerBounds=[], upperBounds=[], varType=[],
                 discreteVals=[], optimum=0.0, pltTitle='', histTitle='',
                 varNames=''):
        """!
        @param self: <em> pointer </em> \n
            The ProblemParameters pointer. \n
        @param lowerBounds: \e array \n
            The lower bounds of the design variable(s). Only enter the bounds
            for continuous and integer/binary variables. The order must match
            the order specified in varType and ub. \n
        @param upperbounds: \e array \n
            The upper bounds of the design variable(s). Only enter the bounds
            for continuous and integer/binary variables. The order must match
            the order specified in varType and lb. \n
        @param varType: <em> list or array </em> \n
            The type of variable for each position in the upper and lower
            bounds array. Discrete variables are to be included last as they
            are specified separatly from the lb/ub throught the discreteVals
            optional input. A variable can have two types (for example, 'dx'
            could denote a layer that can take multiple materials and be placed
            at multiple design locations) \n
            Allowed values: \n
             'c' = continuous over a given range (range specified in lb &
                   ub). \n
             'i' = integer/binary (difference denoted by ub/lb). \n
             'd' = discrete where the allowed values are given by the option
                   discreteVals nxm arrary with n=# of discrete variables and
                   m=# of values that can be taken for each variable. \n
             'x' = combinatorial. All of the variables denoted by x are assumed
                   to be "swappable" in combinatorial permutations.  There must
                   be at least two variables denoted as combinatorial. \n
             'f' = fixed design variable. Will not be considered of any
                   permutation. \n
        @param discreteVals: <em> list of list(s) </em> \n
            nxm with n=# of discrete variables and m=# of values that can be
            taken for each variable. For example, if you had two variables
            representing the tickness and diameter of a cylinder that take
            standard values, the discreteVals could be specified as: \n
            discreteVals = [[0.125, 0.25, 0.375], [0.25, 0.5, 075]] \n
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

        ## @var lb 
        # \e array: The lower bounds of the design variable(s).
        self.lb = lowerBounds

        ## @var ub:
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

    def __repr__(self):
        """!
        ProblemParameters class attribute print function.

        @param self: <em> pointer </em> \n
            The ProblemParameters pointer. \n
        """
        return "ProblemParameters({}, {}, {}, {}, {}, {}, {}, {})".format(
                                                       self.lb,
                                                       self.ub,
                                                       self.varType,
                                                       self.discreteVals,
                                                       self.optimum,
                                                       self.pltTitle,
                                                       self.histTitle,
                                                       self.varNames)

    def __str__(self):
        """!
        Human readable ProblemParameters print function.

        @param self: \e pointer \n
            The ProblemParameters pointer. \n
        """

        header = ["ProblemParameters:"]
        header += ["Lower Bounds: {}".format(self.lb)]
        header += ["Upper Bounds: {}".format(self.ub)]
        header += ["Variable Types: {}".format(self.varType)]
        header += ["Discrete Values: {}".format(self.discreteVals)]
        header += ["Global Optimum: {}".format(self.optimum)]
        header += ["Plot Title: {}".format(self.pltTitle)]
        header += ["Histogram Title: {}".format(self.histTitle)]
        header += ["Variable Names: {}".format(self.varNames)]
        return "\n".join(header)+"\n"

    def get_preset_params(self, funct, algorithm='', dimension=2):
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

        @return <em> ProblemParameters object: </em> A ProblemParameters
            object with populated member variables.
        """

        # Build temp varType array for continuous problems with variable
        # dimensions
        v = []
        for i in range(0, dimension):
            v.append('c')

        for case in Switch(funct):
            if case('mi_spring'):
                params = ProblemParameters([0.01, 1], [3.0, 10],
                                 ['c', 'i', 'd'],
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
                params = ProblemParameters([0.05, 0.25, 2.0], [2.0, 1.3, 15.0],
                                  ['c', 'c', 'c'], [], 0.012665,
                                  ('\\textbf{Spring Optimization using %s}'
                                   %algorithm),
                                  ('\\textbf{Function Evaluations for Spring '
                                   'Optimization using %s}' %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{Width}',
                                  '\\textbf{Diameter}', '\\textbf{Length}'])
                break
            if case('pressure_vessel'):
                params = ProblemParameters([0.0625, 0.0625, 10.0, 1E-8],
                                  [1.25, 99*0.0625, 50.0, 200.0],
                                  ['c', 'c', 'c', 'c'], [], 6059.714335,
                                  ('\\textbf{Pressure Vessel Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for '
                                   'Pressure Vessel Optimization using %s}'
                                    %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{Thickness}',
                                   '\\textbf{Head Thickness}',
                                   '\\textbf{Inner Radius}',
                                   '\\textbf{Cylinder Length}'])
            if case('mi_pressure_vessel'):
                params = ProblemParameters([25.0, 25.0], [210.0, 240.0],
                                  ['c', 'c', 'd', 'd'],
                                  [[0.0625, 0.125, 0.182, 0.25,
                                    0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625,
                                    0.6875, 0.75, 0.7125, 0.875, 0.9375, 1,
                                    1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375,
                                    1.4375, 1.5, 1.5625, 1.625, 1.6875, 1.75,
                                    1.8125, 1.875, 1.9375, 2], [0.0625, 0.125,
                                    0.182, 0.25, 0.3125, 0.375, 0.4375, 0.5,
                                    0.5625, 0.625, 0.6875, 0.75, 0.7125, 0.875,
                                    0.9375, 1, 1.0625, 1.125, 1.1875, 1.25,
                                    1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625,
                                    1.6875, 1.75, 1.8125, 1.875, 1.9375, 2]],
                                  5855.893191,
                                  ('\\textbf{MI Pressure Vessel Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Pressure'
                                   ' Vessel Optimization using %s}'
                                    %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{Radius}',
                                   '\\textbf{Lenght}',
                                   '\\textbf{Shell Thickness}',
                                   '\\textbf{Head Thickness}'])
                break
            if case('welded_beam'):
                params = ProblemParameters([0.1, 0.1, 1E-8, 1E-8],
                                  [10.0, 10.0, 10.0, 2.0], ['c', 'c', 'c', 'c'],
                                  [], 1.724852,
                                  ('\\textbf{Welded Beam Optimization using '
                                   '%s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Welded '
                                   'Beam Optimization using %s}' %algorithm),
                                  ['\\textbf{Fitness}', '\\textbf{Weld H}',
                                   '\\textbf{Weld L}', '\\textbf{Beam H}',
                                   '\\textbf{Beam W}'])
                break
            if case('speed_reducer'):
                params = ProblemParameters([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0],
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
            if case('mi_chemical_proccess'):
                params  = ProblemParameters([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [10.0, 10.0, 10.0, 1, 1, 1, 1],
                                  ['c', 'c', 'c', 'i', 'i', 'i', 'i'], [],
                                  4.579582,
                                  ('\\textbf{MI Chemical Process Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Chemical '
                                   'Process Optimization using %s}' %algorithm),
                                  ['\\textbf{x1}', '\\textbf{x2}',
                                   '\\textbf{x3}', '\\textbf{y1}',
                                   '\\textbf{y2}', '\\textbf{y3}',
                                   '\\textbf{y4}'])
                break
            if case('dejong'):
                params = ProblemParameters(np.ones(dimension)*-5.12,
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
                params = ProblemParameters(np.ones(dimension)*-5.12,
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
                params = ProblemParameters(np.ones(dimension)*-25.,
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
                params = ProblemParameters(np.ones(dimension)*-25.,
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
                params = ProblemParameters(np.array([-100., -100.]),
                                  np.array([100., 100.]), v, [], -1.0000,
                                  ('\\textbf{Easom Function Optimization using '
                                   '%s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Easom '
                                   'Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}'] \
                                   + ['\\textbf{x}', '\\textbf{y}'])
                break
            if case('shifted_easom'):
                params = ProblemParameters(np.array([-100., -100.]),
                                  np.array([100., 100.]), v, [], -1.0000,
                                  ('\\textbf{Shifted Easom Function '
                                   'Optimization using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Shifted '
                                   'Easom Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}'] \
                                   + ['\\textbf{x}', '\\textbf{y}'])
                break
            if case('griewank'):
                params = ProblemParameters(np.ones(dimension)*-600.,
                                  np.ones(dimension)*600., v, [], 0.0000,
                                  ('\\textbf{Griewank Function Optimization '
                                   'using %s}' %algorithm),
                                  ('\\textbf{Function Evaluations for Griewank '
                                   'Function Optimization using %s}'
                                   %algorithm),
                                  ['\\textbf{Fitness}']+['\\textbf{Dim \#%s}' \
                                   %i for i in range(dimension)])
                break
            if case('shifted griewank'):
                params = ProblemParameters(np.ones(dimension)*-600.,
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
                params = ProblemParameters(np.ones(dimension)*-5.12,
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
                params = ProblemParameters(np.ones(dimension)*-5.12,
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
                params = ProblemParameters(np.ones(dimension)*-5.,
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
                params = ProblemParameters(np.ones(dimension)*-5.,
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
                params = ProblemParameters([], [], [], [], 0.0,
                                  ('\\textbf{TSP Optimization using %s}'
                                   %algorithm),
                                  ('\\textbf{Function Evaluations for TSP '
                                   'using %s}' %algorithm),
                                  ['\\textbf{Fitness}'])
                break
            if case():
                print ('ERROR: Fishing in the deep end you are. Define your own'
                       'parameter set you must.')
        return params

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
        if fnew > S.penalty:
            S.penalty=fnew
    # Calculate fitness; replace parents if child has better fitness
    feval=0
    for i in range(0,len(children),1):
        if random_replace:
            j=int(np.random.rand()*len(parents))
        elif len(indices)==len(children):
            j=indices[i]
        else: 
            j=i
        (fnew,gnew)=func(children[i],penalty=S.penalty)
        feval += 1
        if fnew < parents[j].fitness:
            parents[j].fitness=fnew
            parents[j].variables=cp.copy(children[i])
            parents[i].changeCount+=1
            parents[i].stallCount=0
            replace+=1
            if parents[i].changeCount>=25 and j>=S.population*S.fracElite:
                #print "Max Changes: ", parents[i].f
                parents[i].variables=S.initialize(1, 'random', lb, ub, varType).flatten()
                (fnew,gnew)=func(parents[i].variables,penalty=S.penalty)
                parents[i].fitness=fnew
                parents[i].changeCount=0
        else:
            parents[j].stallCount+=1
            if parents[j].stallCount>50000 and j!=0:
                parents[i].variables=S.initialize(1, 'random', lb, ub, varType).flatten()
                (fnew,gnew)=func(parents[i].variables,penalty=S.penalty)
                parents[i].fitness=fnew
                parents[i].changeCount=0
                parents[i].stallCount=0
                       
            # Metropis-Hastings algorithm
            r=int(np.random.rand()*len(parents))
            if r<=mhFrac:
                r=int(np.random.rand()*len(parents))
                if fnew < parents[r].fitness:  
                    parents[r].fitness=fnew
                    parents[r].variables=cp.copy(children[i])
                    parents[r].changeCount+=1
                    parents[r].stallCount+=1
                    replace+=1

    #Sort the pop
    parents.sort(key=lambda x: x.fitness)
    
    # Map the discrete variables for storage
    if discreteID!=[]:
        dVec=[]
        i=0
        for j in range(len(discreteID)):
            if discreteID[j]==1:
                dVec.append(discreteMap[i][int(parents[0].variables[j])])
                i+=1
            else:
                dVec.append(parents[0].variables[j])
    else:
        dVec=cp.copy(parents[0].variables)
        
    #Store history on timeline if new optimal design found
    if len(timeline)<2:
        timeline.append(Event(len(timeline)+1,feval,parents[0].fitness,dVec))
    elif parents[0].fitness<timeline[-1].fitness and abs((timeline[-1].fitness-parents[0].fitness)/parents[0].fitness) > S.convTol:
        timeline.append(Event(timeline[-1].generation+1,timeline[-1].evaluations+feval,parents[0].fitness,dVec))
    else:
        timeline[-1].generation+=genUpdate
        timeline[-1].evaluations+=feval
        
    return(parents,replace,timeline)

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