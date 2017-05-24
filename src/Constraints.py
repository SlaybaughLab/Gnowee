"""!
@file Constraints.py
@package Coeus

@defgroup Constraints Constraints

@brief Defines a class to perform constraint calculations.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""

import numpy as np

from math import ceil, sqrt

#-----------------------------------------------------------------------------#
class Constraint(object):
    """!
    @ingroup Constraints
    The class creates a Constraints object that can be used in
    optimization algorithms.
    """

    def __init__(self, method=None, constraint=None, penalty=1E15):
        """!
        Constructor to build the ObjectiveFunction class.

        @param self: <em> object pointer </em> \n
            The object pointer. \n
        @param method: \e string \n
            The name of the constraint function to evaluate. \n
        @param constraint: \e float \n
            The constraint to be compared against. \n
        @param penalty: \e float \n
            The penalty to be applied if a constraint is violated.  1E15
            is recommended. \n
        """

        ## @var _FUNC_DICT
        # <em> dictionary of function handles: </em> Stores
        # the mapping between the string names and function handles for
        # the constraint function evaluations in the class.  This is a
        # legacy private variable that is only used for error reporting.
        self._FUNC_DICT = {'less_or_equal': self.less_or_equal,
                           'less_than': self.less_than,
                           'greater_than': self.greater_than,
                           'spring': self.spring,
                           'mi_spring': self.mi_spring,
                           'welded_beam': self.welded_beam,
                           'pressure_vessel': self.pressure_vessel,
                           'mi_pressure_vessel': self.mi_pressure_vessel,
                           'speed_reducer': self.speed_reducer,
                           'mi_chemical_process': self.mi_chemical_process}

        ## @var func
        # <em> function handle: </em> The function handle for
        # the constraint function to be used for the optimization.  The
        # function must be specified as a method of the class.
        if method != None and type(method) == str:
            self.set_constraint_func(method)
        else:
            self.func = method

        ## @var constraint
        # \e float: The constraint to be enforced.
        self.constraint = constraint

        ## @var penalty
        # \e float: The penalty to be applied if the constraint
        # is violated
        self.penalty = penalty

    def __repr__(self):
        """!
        Constraint class param print function.

        @param self: \e pointer \n
            The Constraint pointer. \n
        """
        return "Constraint({}, {}, {})".format(self.func.__name__,
                                               self.constraint,
                                               self.penalty)

    def __str__(self):
        """!
        Human readable Constraint print function.

        @param self: \e pointer \n
            The Constraint pointer. \n
        """

        header = ["  Constraint:"]
        header += ["Function: {}".format(self.func.__name__)]
        header += ["Constraint: {}".format(self.constraint)]
        header += ["Penalty: {}".format(self.penalty)]
        return "\n".join(header)+"\n"

    def set_constraint_func(self, funcName):
        """!
        Converts an input string name for a function to a function handle.

        @param self: \e pointer \n
            The Constraint pointer. \n
        @param funcName \e string \n
             A string identifying the constraint function to be used. \n
        """
        if hasattr(funcName, '__call__'):
            self.func = funcName
        else:
            try:
                self.func = getattr(self, funcName)
                assert hasattr(self.func, '__call__'), 'Invalid function handle'
            except KeyError:
                print ('ERROR: The function specified does not exist in the '
                   'Constraints class or the _FUNC_DICT. Allowable methods '
                   'are {}'.format(self._FUNC_DICT))

    def get_penalty(self, violation):
        """!
        Calculate the constraint violation penalty, if any.

        @param self: \e pointer \n
            The Constraint pointer. \n
        @param violation \e float \n
             The magnitude of the constraint violation used for scaling the
             penalty. \n

        @return \e float: The scaled penalty. \n
        """

        return self.penalty*ceil(violation)**2

#-----------------------------------------------------------------------------#
# The following sections are user modifiable to all for the use of new
# objective functions that have not yet been implemented.  The same format must
# be followed to work with the standard Coeus call.
#
# Alternatively, the user can specify additional functions in their own files.
# Examples of both are shown in the runGnowee ipython notebook in the /src
# directory.
#-----------------------------------------------------------------------------#
    def spring(self, u):
        """!
        Spring penalty method of constraint enforcement.

        Optimal Example: \n
        u = [0.05169046, 0.356750, 11.287126] \n
        fitness = 0.0126653101469

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 3, ('Spring design needs to specify D, W, and L and '
                             'only those 3 parameters.')
        assert u[0] != 0 and u[1] != 0 and u[2] != 0, ('Design values {} '
                                                 'cannot be zero.'.format(u))

        # Constraints
        g = [1.-u[1]**3*u[2]/(71785.*u[0]**4)]
        g.append((4*u[1]**2-u[0]*u[1])/(12566*(u[1]*u[0]**3-u[0]**4)) \
                 +1/(5108*u[0]**2)-1)
        g.append(1-140.45*u[0]/(u[1]**2*u[2]))
        g.append((u[0]+u[1])/1.5-1)

        # Evaluate fitness
        totalPenalty = 0
        for constraint in g:
            totalPenalty += self.get_penalty(self.less_or_equal(constraint))
        return totalPenalty

    def mi_spring(self, u):
        """!
        Spring penalty method of constraint enforcement.

        Optimal Example: \n
        u = [1.22304104, 9, 36] = [1.22304104, 9, 0.307]\n
        fitness = 2.65856

        Taken from Lampinen, "Mixed Integer-Discrete-Continuous Optimization
        by Differential Evolution"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n

        @return \e float: The assessed penalty for constraint violations for
            the specified input. \n
        """
        assert len(u) == 3, ('Spring design needs to specify D, N, and d and '
                             'only those 3 parameters.')

        # Set variables
        D = u[0]
        N = u[1]
        d = u[2]

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
        g = [8*Cf*Fmax*D/(np.pi*d**3)-S]
        g.append(lf-lmax)
        g.append(dmin-d)
        g.append(D-Dmax)
        g.append(3.0-D/d)
        g.append(sigmap-sigmapm)
        g.append(sigmap+(Fmax-Fp)/K + 1.05*(N+2)*d-lf)
        g.append(sigmaw-(Fmax-Fp)/K)

        #Evaluate fitness
        totalPenalty = 0
        for constraint in g:
            totalPenalty += self.get_penalty(self.less_or_equal(constraint))
        return totalPenalty

    def welded_beam(self, u):
        """!
        Welded Beam penalty method of constraint enforcement.

        Optimal Example: \n
        u = [0.20572965, 3.47048857, 9.0366249, 0.20572965] \n
        fitness = 1.7248525603892848

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> \n
            The ObjectiveFunction pointer. \n
        @param u: \e array \n
            The design parameters to be evaluated. \n

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 4, ('Welded Beam design needs to specify 4 '
                             'parameters.')
        assert u[0] != 0 and u[1] != 0 and u[2] != 0 and u[3] != 0, ('Design'
                             'values {} cannot be zero'.format(u))

        # Constraints
        em = 6000.*(14+u[1]/2.)
        r = sqrt(u[1]**2/4.+((u[0]+u[2])/2.)**2)
        j = 2.*(u[0]*u[1]*sqrt(2)*(u[1]**2/12.+((u[0]+u[2])/2.)**2))
        tau_p = 6000./(sqrt(2)*u[0]*u[1])
        tau_dp = em*r/j
        tau = sqrt(tau_p**2+2.*tau_p*tau_dp*u[1]/(2.*r)+tau_dp**2)
        sigma = 504000./(u[3]*u[2]**2)
        delta = 65856000./((30*10**6)*u[3]*u[2]**2)
        pc = 4.013*(30.*10**6)*sqrt(u[2]**2*u[3]**6/36.)/196.*(1.-u[2] \
             *sqrt((30.*10**6)/(4.*(12.*10**6)))/28.)

        g = [tau-13600.]
        g.append(sigma-30000.)
        g.append(u[0]-u[3])
        g.append(0.10471*u[0]**2+0.04811*u[2]*u[3]*(14.+u[1])-5.0)
        g.append(0.125-u[0])
        g.append(delta-0.25)
        g.append(6000-pc)

        #Evaluate fitness
        totalPenalty = 0
        for constraint in g:
            totalPenalty += self.get_penalty(self.less_or_equal(constraint))
        return totalPenalty

    def pressure_vessel(self, u):
        """!
        Pressure vessel penalty method of constraint enforcement.

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

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 4, ('Pressure vesseldesign needs to specify 4 '
                             'parameters.')
        assert u[0] != 0 and u[1] != 0 and u[2] != 0 and u[3] != 0, ('Design'
                             'values {} cannot be zero'.format(u))

        # Constraints
        g = [-u[0]+0.0193*u[2]]
        g.append(-u[1]+0.00954*u[2])
        g.append(-np.pi*u[2]**2*u[3]-4./3.*np.pi*u[2]**3+1296000)
        g.append(u[3]-240)

        #Evaluate fitness
        totalPenalty = 0
        for constraint in g:
            totalPenalty += self.get_penalty(self.less_or_equal(constraint))
        return totalPenalty

    def mi_pressure_vessel(self, u):
        """!
        Mixed Integer Pressure vessel penalty method of constraint enforcement.

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

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 4, ('MI Pressure vessel design needs to specify 4 '
                             'parameters.')

        # Set variables
        R = u[0]
        L = u[1]
        ts = u[2]
        th = u[3]

        # Constraints
        g = [-ts+0.01932*R]
        g.append(-th+0.00954*R)
        g.append(-np.pi*R**2*L-4.0/3.0*np.pi*R**3+750*1728)
        g.append(-240+L)

        #Evaluate fitness
        totalPenalty = 0
        for constraint in g:
            totalPenalty += self.get_penalty(self.less_or_equal(constraint))
        return totalPenalty

    def speed_reducer(self, u):
        """!
        Speed reducer penalty method of constraint enforcement.

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
        g = [27./(u[0]*u[1]**2*u[2])-1.]
        g.append(397.5/(u[0]*u[1]**2*u[2]**2)-1.)
        g.append(1.93*u[3]**3/(u[1]*u[2]*u[5]**4)-1.)
        g.append(1.93*u[4]**3/(u[1]*u[2]*u[6]**4)-1.)
        g.append(1.0/(110.*u[5]**3)*sqrt((745.0*u[3]/(u[1]*u[2]))**2\
                                           +16.9*10**6)-1)
        g.append(1.0/(85.*u[6]**3)*sqrt((745.0*u[4]/(u[1]*u[2]))**2\
                                          +157.5*10**6)-1)
        g.append(u[1]*u[2]/40.-1)
        g.append(5.*u[1]/u[0]-1)
        g.append(u[0]/(12.*u[1])-1)
        g.append((1.5*u[5]+1.9)/u[3]-1)
        g.append((1.1*u[6]+1.9)/u[4]-1)

        #Evaluate fitness
        totalPenalty = 0
        for constraint in g:
            totalPenalty += self.get_penalty(self.less_or_equal(constraint))
        return totalPenalty

    def mi_chemical_process(self, u):
        """!
        Chemical process design constraint enforcement.

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

        @return \e array: The fitness associated with the specified input. \n
        @return \e array: The assessed value for each constraint for the
            specified input. \n
        """
        assert len(u) == 7, ('Chemical process design needs to specify 7 '
                             'parameters.')

        # Constraints
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
        totalPenalty = 0
        for constraint in g:
            totalPenalty += self.get_penalty(self.less_or_equal(constraint))
        return totalPenalty

    def less_or_equal(self, candidate):
        """!
        Compares a previously calculated value to a user specifed maximum
        including that maximum.

        @param self: \e pointer \n
            The Constraint pointer. \n
        @param candidate \e float \n
             The calculated value corresponding to a candidate design.\n

        @return \e float: The penalty associated with the candidate design. \n
        """
        if candidate <= self.constraint:
            return 0
        else:
            return self.get_penalty(candidate-self.constraint)

    def less_than(self, candidate):
        """!
        Compares a previously calculated value to a user specifed maximum
        excluding that maximum.

        @param self: \e pointer \n
            The Constraint pointer. \n
        @param candidate \e float \n
             The calculated value corresponding to a candidate design.\n

        @return \e float: The penalty associated with the candidate design. \n
        """

        if candidate < self.constraint:
            return 0
        else:
            return self.get_penalty(candidate-self.constraint)

    def greater_than(self, candidate):
        """!
        Compares the calculated value to the minimum specified by the user.

        @param self: \e pointer \n
            The Constraint pointer. \n
        @param candidate \e float \n
             The calculated value corresponding to a candidate design.\n

        @return \e float: The penalty associated with the candidate design. \n
        """

        if candidate > self.constraint:
            return 0
        else:
            return self.get_penalty(self.constraint - candidate)
