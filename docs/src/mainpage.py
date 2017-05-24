"""!
\mainpage

# Gnowee

\version 1.0

Gnowee is a general purpose hybrid metaheuristic optimization algorithm designed for rapid convergence to nearly globally optimum solutions for complex, constrained engineering problems with mixed-integer and combinatorial design vectors and high-cost, noisy, discontinuous, black box objective function evaluations. Gnowee's hybrid metaheuristic framework is based on a set of diverse, robust heuristics that appropriately balance diversification and intensification strategies across a wide range of optimization problems.
Comparisons between Gnowee and several well-established metaheuristic algorithms are made for a set of eighteen continuous, mixed-integer, and combinatorial benchmarks. A summary of these benchmarks is <a href='../Benchmarks/results/Gnowee_Benchmark_Results.pdf'>available</a>. These results demonstrate Gnoweee to have superior flexibility and convergence characteristics over a wide range of design spaces.

A paper, describing the Gnowee framework and benchmarks is <a href='../IEEE_Gnowee.pdf'>available</a>

## Running Gnowee

For examples on how to run Gnowee, please refer to the <a href='../../src/runGnowee.ipynb'>runGnowee notebook</a> included in the <a href='../../src'>src directory</a>.  This contains multiple examples of how to modify and run Gnowee.

## Building Documentation

To build the documentation, in the <a href='../src'>docs/src directory</a> run the command:

\>> doxygen Doxyfile

This will build the html and latex version of the documentation.  The <a href='../GnoweeDocs.html'>symlink</a> in the <a href='../'>docs directory</a> for the html index should automatically update.  If not the html index can be found <a href='../html/index.html'>here</a>.

The up-to-date latex documentation is included in <a href='../GnoweeDocs.pdf'> pdf form</a>.  If an update of the latex documentation is desired, go to the <a href='../docs/latex'>docs/latex directory </a> and run the command:

\>> make

This will build the latex documentation.  The updated documentation file will be named <a href='../docs/latex/refman.pdf'>refman.pdf </a> and be placed in this directory.


## Citation Information
To cite Gnowee, use the following: \n


## Contact information

Bugs and suggestions for improvement can be submitted via the GitHub page:
https://github.com/SlaybaughLab/Gnowee

Alternatively, questions or comments on Gnowee can be directed to: \n
James Bevins \n
james.e.bevins@gmail.com \n

## Licensing Information
\copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC Berkeley Copyright and Disclaimer Notice</a>
\license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>

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

## Acknowledements
This material is based upon work supported by the National Science Foundation
Graduate Research Fellowship under Grant No. NSF 11-582.
"""