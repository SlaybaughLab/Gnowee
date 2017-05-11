"""!
@file src/ObjectiveFunction.py
@package Gnowee

@defgroup ObjectiveFunction ObjectiveFunction

@brief Defines a class to perform objective function calculations.

This class contains the necessary functions and methods to create objective
functions and initialize the necessary parameters. The class is pre-stocked
with common benchmark functions for easy fishing.

Users can modify the this class to add additional functions following the
format of the functions currently in the class.

@author James Bevins

@date 10May17
"""

import math as m
import numpy as np
import operator

from Constraints import Constraint
from GnoweeUtilities import ProblemParameters

#-----------------------------------------------------------------------------#
class ObjectiveFunction(object):
    """!
    @ingroup ObjectiveFunction
    The class creates a ObjectiveFunction object that can be used in
    optimization algorithms.
    """

    def __init__(self, method=None, objective=None):
        """!
        Constructor to build the ObjectiveFunction class.

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param method: \e string \n
            The name of the objective function to evaluate. \n
        @param objective: <em> integer, float, or numpy array </em> \n
            The desired objective associated with the optimization.  The
            chosen value and type must be compatible with the optiization
            function chosen. \n
        """

        ## @var _FUNC_DICT
        # <em> dictionary of function handles </em> Stores
        # the mapping between the string names and function handles for
        # the objective function evaluations in the class.  This must be
        # updated by the user if a function is added to the class.
        self._FUNC_DICT = {'spring': self.spring,
                           'mi_spring': self.mi_spring,
                           'welded_beam': self.welded_beam,
                           'pressure_vessel': self.pressure_vessel,
                           'mi_pressure_vessel': self.mi_pressure_vessel,
                           'speed_reducer': self.speed_reducer,
                           'mi_chemical_process': self.mi_chemical_process,
                           'ackley': self.ackley,
                           'shifted_ackley': self.shifted_ackley,
                           'dejong': self.dejong,
                           'shifted_dejong': self.shifted_dejong,
                           'easom': self.easom,
                           'shifted_easom': self.shifted_easom,
                           'griewank': self.griewank,
                           'shifted_griewank': self.shifted_griewank,
                           'rastrigin': self.rastrigin,
                           'shifted_rastrigin': self.shifted_rastrigin,
                           'rosenbrock': self.rosenbrock,
                           'shifted_rosenbrock': self.shifted_rosenbrock,
                           'tsp': self.tsp}

        ## @var func
        # <em> function handle </em> The function handle for the
        # objective function to be used for the optimization.  The
        # function must be specified as a method of the class.
        if method != None and type(method) == str:
            self.set_obj_func(method)
        else:
            self.func = method

        ## @var objective
        # <em> integer, float, or numpy array </em> The desired outcome
        # of the optimization.
        self.objective = objective

    def __repr__(self):
        """!
        ObjectiveFunction class param print function.

        @param self: \e pointer \n
            The ObjectiveFunction pointer. \n
        """
        return "ObjectiveFunction({}, {}, {})".format(self._FUNC_DICT,
                                                      self.func.__name__,
                                                      self.objective)

    def __str__(self):
        """!
        Human readable ObjectiveFunction print function.

        @param self: \e pointer \n
            The ObjectiveFunction pointer. \n
        """

        header = ["ObjectiveFunction:"]
        header += ["Objective Function Dictionary: {}".format(
                   self._FUNC_DICT)]
        header += ["Objective Functions: {}".format(self.func.__name__)]
        header += ["Objective: {}".format(self.objective)]
        return "\n".join(header)+"\n"

    def set_obj_func(self, funcName):
        """!
        Converts an input string name for a function to a function handle.

        @param self: \e pointer \n
            The ObjectiveFunction pointer. \n
        @param funcName \e string \n
             A string identifying the objective function to be used. \n
        """
        try:
            self.func = self._FUNC_DICT[funcName]
            assert hasattr(self.func, '__call__'), 'Invalid function handle'
        except KeyError:
            print ('ERROR: The function specified does not exist in the '
                   'ObjectiveFunction class or the _FUNC_DICT. Allowable '
                   'methods are {}'.format(self._FUNC_DICT))

#-----------------------------------------------------------------------------#
# The following sections are user modifiable to all for the use of new
# objective functions that have not yet been implemented.  The same format must
# be followed to work with the standard Coeus call. If a function is added.
# it must also be added to the _FUNC_DICT attribute of the class.
#-----------------------------------------------------------------------------#
    def spring(self, u, penalty=1E15):
        """!
        Spring objective function with penalty method of constraint
        enforcement.

        Optimal Example: \n
        u = [0.05169046, 0.356750, 11.287126] \n
        fitness = 0.0126653101469

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 3, ('Spring design needs to specify D, W, and L and '
                             'only those 3 parameters.')
        assert u[0] != 0 and u[1] != 0 and u[2] != 0, ('Design values {} '
                                                 'cannot be zero.'.format(u))

        # Constraints
        ineqConstraint = Constraint(method='less_or_equal', constraint=0.0,
                                    penalty=penalty)
        g = [1.-u[1]**3*u[2]/(71785.*u[0]**4)]
        g.append((4*u[1]**2-u[0]*u[1])/(12566*(u[1]*u[0]**3-u[0]**4)) \
                 +1/(5108*u[0]**2)-1)
        g.append(1-140.45*u[0]/(u[1]**2*u[2]))
        g.append((u[0]+u[1])/1.5-1)

        # Evaluate fitness
        fitness = ((2+u[2])*u[0]**2*u[1])
        for constraint in g:
            fitness += ineqConstraint.get_penalty(ineqConstraint.func(
                                                                 constraint))
        return fitness, g

    def mi_spring(self, u, penalty=1E15):
        """!
        Spring objective function with penalty method of constraint
        enforcement.

        Optimal Example: \n
        u = [1.22304104, 9, 36] = [1.22304104, 9, 0.307]\n
        fitness = 2.65856

        Taken from Lampinen, "Mixed Integer-Discrete-Continuous Optimization
        by Differential Evolution"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 3, ('Spring design needs to specify D, N, and d and '
                             'only those 3 parameters.')

        # Set variables
        tmp = ProblemParameters()
        tmp.set_preset_params('mi_spring', 'Gnowee')
        diams = tmp.discreteVals[0]

        D = u[0]
        N = u[1]
        d = diams[int(u[2])]

        # Variable Definititions:
        Fmax = 1000
        S = 189000.0
        Fp = 300
        sigmapm = 6.0
        sigmaw = 1.25
        G = 11.5*10**6
        lmax = 14
        dmin = 0.2
        Dmax = 3.0
        K = G*d**4/(8*N*D**3)
        sigmap = Fp/K
        Cf = (4*(D/d)-1)/(4*(D/d)-4)+0.615*d/D
        lf = Fmax/K+1.05*(N+2)*d

        # Constraints
        ineqConstraint = Constraint(method='less_or_equal', constraint=0.0,
                                    penalty=penalty)
        g = [8*Cf*Fmax*D/(np.pi*d**3)-S]
        g.append(lf-lmax)
        g.append(dmin-d)
        g.append(D-Dmax)
        g.append(3.0-D/d)
        g.append(sigmap-sigmapm)
        g.append(sigmap+(Fmax-Fp)/K + 1.05*(N+2)*d-lf)
        g.append(sigmaw-(Fmax-Fp)/K)

        #Evaluate fitness
        fitness = np.pi**2*D*d**2*(N+2)/4
        for constraint in g:
            fitness += ineqConstraint.get_penalty(ineqConstraint.func(
                                                                 constraint))
        return fitness, g

    def welded_beam(self, u, penalty=1E15):
        """!
        Welded Beam objective function with penalty method of constraint
        enforcement.

        Optimal Example: \n
        u = [0.20572965, 3.47048857, 9.0366249, 0.20572965] \n
        fitness = 1.7248525603892848

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 4, ('Welded Beam design needs to specify 4 '
                             'parameters.')
        assert u[0] != 0 and u[1] != 0 and u[2] != 0 and u[3] != 0, ('Design'
                             'values {} cannot be zero'.format(u))

        # Constraints
        ineqConstraint = Constraint(method='less_or_equal', constraint=0.0,
                                    penalty=penalty)
        em = 6000.*(14+u[1]/2.)
        r = m.sqrt(u[1]**2/4.+((u[0]+u[2])/2.)**2)
        j = 2.*(u[0]*u[1]*m.sqrt(2)*(u[1]**2/12.+((u[0]+u[2])/2.)**2))
        tau_p = 6000./(m.sqrt(2)*u[0]*u[1])
        tau_dp = em*r/j
        tau = m.sqrt(tau_p**2+2.*tau_p*tau_dp*u[1]/(2.*r)+tau_dp**2)
        sigma = 504000./(u[3]*u[2]**2)
        delta = 65856000./((30*10**6)*u[3]*u[2]**2)
        pc = 4.013*(30.*10**6)*m.sqrt(u[2]**2*u[3]**6/36.)/196.*(1.-u[2] \
             *m.sqrt((30.*10**6)/(4.*(12.*10**6)))/28.)

        g = [tau-13600.]
        g.append(sigma-30000.)
        g.append(u[0]-u[3])
        g.append(0.10471*u[0]**2+0.04811*u[2]*u[3]*(14.+u[1])-5.0)
        g.append(0.125-u[0])
        g.append(delta-0.25)
        g.append(6000-pc)

        #Evaluate fitness
        fitness = 1.10471*u[0]**2*u[1]+0.04811*u[2]*u[3]*(14.0+u[1])
        for constraint in g:
            fitness += ineqConstraint.get_penalty(ineqConstraint.func(
                                                                 constraint))
        return fitness, g

    def pressure_vessel(self, u, penalty=1E15):
        """!
        Pressure vessel objective function with penalty method of constraint
        enforcement.

        Near Optimal Example: \n
        u = [0.81250000001, 0.4375, 42.098445595854923, 176.6365958424394] \n
        fitness = 6059.714335

        Optimal obtained using Gnowee: \n
        u = [0.7781686880924992, 0.3846491857203429, 40.319621144688995,
             199.99996630362293] \n
        fitness = 5885.33285347

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 4, ('Pressure vesseldesign needs to specify 4 '
                             'parameters.')
        assert u[0] != 0 and u[1] != 0 and u[2] != 0 and u[3] != 0, ('Design'
                             'values {} cannot be zero'.format(u))

        # Constraints
        ineqConstraint = Constraint(method='less_or_equal', constraint=0.0,
                                    penalty=penalty)
        g = [-u[0]+0.0193*u[2]]
        g.append(-u[1]+0.00954*u[2])
        g.append(-np.pi*u[2]**2*u[3]-4./3.*np.pi*u[2]**3+1296000)
        g.append(u[3]-240)

        #Evaluate fitness
        fitness = 0.6224*u[0]*u[2]*u[3]+1.7781*u[1]*u[2]**2+3.1661*u[0]**2 \
                  *u[3]+19.84*u[0]**2*u[2]
        for constraint in g:
            fitness += ineqConstraint.get_penalty(ineqConstraint.func(
                                                                 constraint))
        return fitness, g

    def mi_pressure_vessel(self, u, penalty=1E15):
        """!
        Mixed Integer Pressure vessel objective function with penalty method
        of constraint enforcement.

        Near optimal example: \n
        u = [58.2298, 44.0291, 17, 9] \n
        fitness = 7203.24

        Optimal example obtained with Gnowee: \n
        u = [38.819876, 221.985576, 0.750000, 0.375000] \n
        fitness = 5855.893191

        Taken from: "Nonlinear Integer and Discrete Programming in Mechanical
        Design Optimization"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 4, ('MI Pressure vessel design needs to specify 4 '
                             'parameters.')

        # Set variables
        tmp = ProblemParameters()
        tmp.set_preset_params('mi_pressure_vessel', 'Gnowee')
        thickness = tmp.discreteVals[0]
        R = u[0]
        L = u[1]
        ts = thickness[int(u[2])]
        th = thickness[int(u[3])]

        # Constraints
        ineqConstraint = Constraint(method='less_or_equal', constraint=0.0,
                                    penalty=penalty)
        g = [-ts+0.01932*R]
        g.append(-th+0.00954*R)
        g.append(-np.pi*R**2*L-4.0/3.0*np.pi*R**3+750*1728)
        g.append(-240+L)

        #Evaluate fitness
        fitness = 0.6224*R*ts*L+1.7781*R**2*th+3.1611*ts**2*L+19.8621*R*ts**2
        for constraint in g:
            fitness += ineqConstraint.get_penalty(ineqConstraint.func(
                                                                 constraint))
        return fitness, g

    def speed_reducer(self, u, penalty=1E15):
        """!
        Speed reducer objective function with penalty method of constraint
        enforcement.

        Optimal example: \n
        u = [58.2298, 44.0291, 17, 9] \n
        fitness = 2996.34784914

        Optimal example obtained with Gnowee: \n
        u = [3.500000, 0.7, 17, 7.300000, 7.800000, 3.350214, 5.286683] \n
        fitness = 5855.893191

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 7, ('Speed reducer design needs to specify 7 '
                             'parameters.')
        assert u[0] != 0 and u[1] != 0 and u[2] != 0 and u[3] != 0 and \
               u[4] != 0 and u[5] != 0 and u[6] != 0, ('Design values cannot '
                                                      'be zero {}.'.format(u))

        # Constraints
        ineqConstraint = Constraint(method='less_or_equal', constraint=0.0,
                                    penalty=penalty)
        g = [27./(u[0]*u[1]**2*u[2])-1.]
        g.append(397.5/(u[0]*u[1]**2*u[2]**2)-1.)
        g.append(1.93*u[3]**3/(u[1]*u[2]*u[5]**4)-1.)
        g.append(1.93*u[4]**3/(u[1]*u[2]*u[6]**4)-1.)
        g.append(1.0/(110.*u[5]**3)*m.sqrt((745.0*u[3]/(u[1]*u[2]))**2\
                                           +16.9*10**6)-1)
        g.append(1.0/(85.*u[6]**3)*m.sqrt((745.0*u[4]/(u[1]*u[2]))**2\
                                          +157.5*10**6)-1)
        g.append(u[1]*u[2]/40.-1)
        g.append(5.*u[1]/u[0]-1)
        g.append(u[0]/(12.*u[1])-1)
        g.append((1.5*u[5]+1.9)/u[3]-1)
        g.append((1.1*u[6]+1.9)/u[4]-1)

        #Evaluate fitness
        fitness = 0.7854*u[0]*u[1]**2*(3.3333*u[2]**2+14.9334*u[2]-43.0934) \
                  - 1.508*u[0]*(u[5]**2+u[6]**2) + 7.4777*(u[5]**3+u[6]**3) \
                  + 0.7854*(u[3]*u[5]**2+u[4]*u[6]**2)
        for constraint in g:
            fitness += ineqConstraint.get_penalty(ineqConstraint.func(
                                                                 constraint))
        return fitness, g

    def mi_chemical_process(self, u, penalty=1E15):
        """!
        Chemical process design mixed integer problem.

        Optimal example: \n
        u = [(0.2, 0.8, 1.907878, 1, 1, 0, 1] \n
        fitness = 4.579582

        Taken from: "An Improved PSO Algorithm for Solving Non-convex
        NLP/MINLP Problems with Equality Constraints"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated.
            [x1, x2, x3, y1, y2, y3, y4] \n
        @param penalty: \e float \n
            Per constraint violation penalty. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 7, ('Chemical process design needs to specify 7 '
                             'parameters.')

        # Constraints
        ineqConstraint = Constraint(method='less_or_equal', constraint=0.0,
                                    penalty=penalty)
        g = [u[0]+u[1]+u[2]+u[3]+u[4]+u[5]-5]
        g.append(u[0]**2+u[1]**2+u[2]**2+u[5]**2-5.5)
        g.append(u[0]+u[3]-1.2)
        g.append(u[1]+u[4]-1.8)
        g.append(u[2]+u[5]-2.5)
        g.append(u[0]+u[6]-1.2)
        g.append(u[1]**2+u[4]**2-1.64)
        g.append(u[2]**2+u[5]**2-4.25)
        g.append(u[2]**2+u[4]**2-4.64)

        #Evaluate fitness
        fitness = (u[3]-1)**2 + (u[4]-2)**2 + (u[5]-1)**2 - m.log(u[6]+1) \
                  + (u[0]-1)**2 + (u[1]-2)**2 + (u[2]-3)**2
        for constraint in g:
            fitness += ineqConstraint.get_penalty(ineqConstraint.func(
                                                                 constraint))
        return fitness, g

    def ackley(self, u, penalty=0.0):
        """!
        Ackley Function: Mulitmodal, n dimensional

        Optimal example: \n
        u = [0, 0, 0, 0, ... n-1] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1, ('The Ackley Function must have a '
                             'dimension greater than 1.')

        #Evaluate fitness
        fitness = -20*m.exp(-1./5.*m.sqrt(1./len(u) \
                                    *sum(u[i]**2 for i in range(len(u))))) \
                - m.exp(1./len(u)*sum(m.cos( \
                        2*m.pi*u[i]) for i in range(len(u)))) + 20 + m.exp(1)

        return fitness, 0.0

    def shifted_ackley(self, u, penalty=0.0):
        """!
        Ackley Function: Mulitmodal, n dimensional
        Ackley Function that is shifted from the symmetric 0, 0, 0, ..., 0
        optimimum.

        Optimal example: \n
        u = [0, 1, 2, 3, ... n-1] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1, ('The Shifted Ackley Function must have a '
                             'dimension greater than 1.')

        #Evaluate fitness
        fitness = -20*m.exp(-1./5.*m.sqrt(1./len(u) \
                                  *sum((u[i]-i)**2 for i in range(len(u))))) \
                - m.exp(1./len(u)*sum(m.cos(2*m.pi* \
                             (u[i]-i)) for i in range(len(u)))) + 20 + m.exp(1)

        return fitness, 0.0

    def dejong(self, u, penalty=0.0):
        """!
        De Jong Function: Unimodal, n-dimensional

        Optimal example: \n
        u = [0, 0, 0, 0, ... n-1] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1, ('The De Jong Function must have a '
                             'dimension greater than 1.')

        #Evaluate fitness
        fitness = sum(i**2 for i in u)

        return fitness, 0.0

    def shifted_dejong(self, u, penalty=0.0):
        """!
        De Jong Function: Unimodal, n-dimensional
        De Jong Function that is shifted from the symmetric 0, 0, 0, ..., 0
        optimimum.

        Optimal example: \n
        u = [0, 1, 2, 3, ... n-1] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1, ('The Shifted De Jong Function must have a '
                             'dimension greater than 1.')

        #Evaluate fitness
        fitness = sum((u[i]-i)**2 for i in range(len(u)))

        return fitness, 0.0

    def easom(self, u, penalty=0.0):
        """!
        Easom Function: Multimodal, n-dimensional

        Optimal example: \n
        u = [pi, pi] \n
        fitness = 1.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 2, 'The Easom Function must have a dimension of 2.'

        #Evaluate fitness
        fitness = -m.cos(u[0])*m.cos(u[1])*m.exp(-(u[0]-m.pi)**2 \
                                                 -(u[1]-m.pi)**2)
        return fitness, 0.0

    def shifted_easom(self, u, penalty=0.0):
        """!
        Easom Function: Multimodal, n-dimensional
        Easom Function that is shifted from the symmetric pi, pi optimimum.

        Optimal example: \n
        u = [pi, pi+1] \n
        fitness = 1.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 2, 'The Easom Function must have a dimension of 2.'

        #Evaluate fitness
        fitness = -m.cos(u[0])*m.cos(u[1]-1)*m.exp(-(u[0]-m.pi)**2 \
                                                 -(u[1]-1-m.pi)**2)
        return fitness, 0.0

    def griewank(self, u, penalty=0.0):
        """!
        Griewank Function: Multimodal, n-dimensional

        Optimal example: \n
        u = [0, 0, 0, ..., 0] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1 and len(u) <= 600, ('The Shifted Griewank Function '
                                   'must have a dimension between 1 and 600.')

        #Evaluate fitness
        fitness = 1./4000.*sum((u[i])**2 for i in range(len(u))) \
                  - prod(m.cos(u[i]/m.sqrt(i+1)) for i in range(len(u))) +1.
        return fitness, 0.0

    def shifted_griewank(self, u, penalty=0.0):
        """!
        Griewank Function: Multimodal, n-dimensional
        Griewank Function that is shifted from the symmetric 0, 0, 0, ..., 0
        optimimum.

        Optimal example: \n
        u = [0, 1, 2, ..., n-1] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1 and len(u) <= 600, ('The Shifted Griewank Function '
                                   'must have a dimension between 1 and 600.')

        #Evaluate fitness
        fitness = 1./4000.*sum((u[i]-i)**2 for i in range(len(u))) \
                  -prod(m.cos((u[i]-i)/m.sqrt(i+1)) for i in range(len(u))) +1.
        return fitness, 0.0

    def rastrigin(self, u, penalty=0.0):
        """!
        Rastrigin Function: Multimodal, n-dimensional

        Optimal example: \n
        u = [0, 0, 0, ..., 0] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1, ('The Rastrigin Function must have a '
                             'dimension greater than 1.')

        #Evaluate fitness
        fitness = 10.*len(u)+sum((u[i])**2 -10. \
                               *np.cos(2.*np.pi*u[i]) for i in range(len(u)))
        return fitness, 0.0

    def shifted_rastrigin(self, u, penalty=0.0):
        """!
        Rastrigin Function: Multimodal, n-dimensional
        Rastrigin Function that is shifted from the symmetric 0, 0, 0, ..., 0
        optimimum.

        Optimal example: \n
        u = [0, 1, 2, ..., n-1] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1, ('The Shifted Rastrigin Function must have a '
                             'dimension greater than 1.')

        #Evaluate fitness
        fitness = 10.*len(u)+sum((u[i]-i)**2 -10. \
                           *np.cos(2.*np.pi*(u[i]-i)) for i in range(len(u)))
        return fitness, 0.0

    def rosenbrock(self, u, penalty=0.0):
        """!
        Rosenbrock Function: uni-modal, n-dimensional.

        Optimal example: \n
        u = [1, 1, 1, ..., 1] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1, ('The Rosenbrock Function must have a '
                             'dimension greater than 1.')

        #Evaluate fitness
        fitness = sum((u[i]-1)**2 +100. \
                    *(u[i+1]-u[i]**2)**2 for i in range(len(u)-1))
        return fitness, 0.0

    def shifted_rosenbrock(self, u, penalty=0.0):
        """!
        Rosenbrock Function: uni-modal, n-dimensional
        Rosenbrock Function that is shifted from the symmetric 0,0,0...0
        optimimum.

        Optimal example: \n
        u = [1, 2, 3, ...n] \n
        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) >= 1, ('The Shifted Rosenbrock Function must have a '
                             'dimension greater than 1.')

        #Evaluate fitness
        fitness = sum((u[i]-1-i)**2 +100.*((u[i+1]-(i+1)) \
                                    -(u[i]-i)**2)**2 for i in range(len(u)-1))
        return fitness, 0.0

    def tsp(self, u, penalty=0.0):
        """!
        Generic objective funtion to evaluate the TSP optimization by
        calculating total distance traveled.

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The city pairs to be evaluated. \n
        @param penalty: \e float \n
            Per constraint violation penalty. Not used for this function;
            included for method input consistency. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 7, ('Speed reducer design needs to specify 4 '
                             'parameters.')

        #Evaluate fitness
        fitness = 0
        for i in range(1, len(u), 1):
            fitness = fitness+round(m.sqrt((u[i][0]-u[i-1][0])**2 \
                                           +(u[i][1]-u[i-1][1])**2))

        #Complete the tour
        fitness = fitness+round(m.sqrt((u[0][0]-u[-1][0])**2 \
                                       +(u[0][1]-u[-1][1])**2))
        return fitness, 0.0

#-----------------------------------------------------------------------------#
def prod(iterable):
    """!
    @ingroup ObjectiveFunction
    Computes the product of a set of numbers (ie big PI, mulitplicative
    equivalent to sum).

    @param iterable: <em> list or array or generator </em>
        Iterable set to multiply.

    @return \e float: The product of all of the items in iterable
    """
    return reduce(operator.mul, iterable, 1)
