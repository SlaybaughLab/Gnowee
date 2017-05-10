"""!
@file Constraints.py
@package Coeus

@defgroup Constraints Constraints

@brief Defines a class to perform constraint calculations.

@author James Bevins,

@date 9May17
"""

from math import ceil

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
        # the constraint function evaluations in the class.  This must be
        # updated by the user if a function is added to the class.
        self._FUNC_DICT = {"less_or_equal": self.less_or_equal,
                           "less_than": self.less_than,
                           "greater_than": self.greater_than}

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
        return "Constraint({}, {}, {}, {})".format(self._FUNC_DICT,
                                                   self.func.__name__,
                                                   self.constraint,
                                                   self.penalty)

    def __str__(self):
        """!
        Human readable Constraint print function.

        @param self: \e pointer \n
            The Constraint pointer. \n
        """

        header = ["Constraint:"]
        header += ["Constraint Function Dictionary: {}".format(
                   self._FUNC_DICT)]
        header += ["Constraint Function: {}".format(self.func.__name__)]
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
        try:
            self.func = self._FUNC_DICT[funcName]
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
# be followed to work with the standard Coeus call. If a function is added.
# it must also be added to the _FUNC_DICT attribute of the class.
#-----------------------------------------------------------------------------#

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
