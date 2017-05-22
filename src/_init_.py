"""!
@package Gnowee

@brief Contains the Gnowee optimization program and associated utilities.

@version 1.0

General nearly-global metaheuristic optimization algorithm. Uses a blend of
common heuristics to solve difficult gradient free constrained MINLP problems
with categorical variables. It is capable of solving  simpler problems, but may
not be the algorithm of choice.

For examples on how to run Gnowee, please refer to the runGnowee notebook
included in the src directory.

@author James Bevins

@date 9May17

\sa Gnowee
\sa GnoweeHeuristics
\sa GnoweeUtilities
\sa ObjectiveFunction
\sa Constraints
\sa OptiPlot
\sa Sampling
\sa ExampleFunction
"""

__all__ = ["Gnowee", "Gnoweeheuristics", "GnoweeUtilities", "Constraints",
           "ObjectiveFunction", "OptiPlot", "Sampling", "ExampleFunction"]
