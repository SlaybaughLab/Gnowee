"""!
@package Gnowee

@brief Contains the Gnowee optimization program and associated utilities.

@version 1.0

Gnowee is a general purpose hybrid metaheuristic optimization algorithm designed for rapid convergence to nearly globally optimum solutions for complex, constrained engineering problems with mixed-integer and combinatorial design vectors and high-cost, noisy, discontinuous, black box objective function evaluations. Gnowee's hybrid metaheuristic framework is based on a set of diverse, robust heuristics that appropriately balance diversification and intensification strategies across a wide range of optimization problems.
Comparisons between Gnowee and several well-established metaheuristic algorithms are made for a set of eighteen continuous, mixed-integer, and combinatorial benchmarks. A summary of these benchmarks is <a href='../Benchmarks/results/Gnowee_Benchmark_Results.pdf'>available</a>. These results demonstrate Gnoweee to have superior flexibility and convergence characteristics over a wide range of design spaces.

A paper, describing the Gnowee framework and benchmarks is <a href='../docs/IEEE_Gnowee.pdf'>available</a>

For examples on how to run Gnowee, please refer to the
<a href='runGnowee.ipynb'>runGnowee ipython notebook </a> included in
the <a href='../src/'>src directory</a>.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>

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
