"""!
@file src/ExampleFunction.py
@package Gnowee

@defgroup ExampleFunction ExampleFunction

@brief Example function to show how user specified functions can be used with
Gnowee.

Several standard optimization benchmarks are provided in the Constraints and
ObjectiveFunction classes, but users are free to  specify and import any
constraints or objective functions desired to use with Gnowee. This module,
along with the example case in the runGnowee piython notebook illustrate how
this is done.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""

#------------------------------------------------------------------------------#
def spring(u):
    """!
    @ingroup ExampleFunction
    Spring objective function.

    Nearly optimal Example: \n
    u = [0.05169046, 0.356750, 11.287126] \n
    fitness = 0.0126653101469

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
    assert len(u) == 3, ('Spring design needs to specify D, W, and L and '
                         'only those 3 parameters.')
    assert u[0] != 0 and u[1] != 0 and u[2] != 0, ('Design values {} '
                                                   'cannot be zero.'.format(u))

    # Evaluate fitness
    fitness = ((2+u[2])*u[0]**2*u[1])

    return fitness
