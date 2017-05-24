"""!
@file src/TSP.py
@package Gnowee

@defgroup TSP TSP

@brief Defines a class to perform Travelling Salesman Problem (TSP)
optimization.

This class is designed to initialize and store TSP problems from the TSPLIB
database. It will read in standard TSPLIB files, and create a TSP object
for use in optimization routines.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""

import numpy as np

from ObjectiveFunction import ObjectiveFunction

#-----------------------------------------------------------------------------#
class TSP(object):
    """!
    @ingroup TSP
    This class creates a TSP object that can be used in optimization
    algorithms to solve the Travelling Saleman Problem.
    """

    def __init__(self, name='', dimension=1, nodes=[], optimum=0.0):
        """!
        Constructor for the TSP class.

        @param self: <em> TSP pointer </em> \n
            The TSP pointer. \n
        @param name: \e string \n
            The name of the TSPLIB problem. \n
        @param dimension: \e integer \n
            The number of nodes (cities) in the problem. \n
        @param nodes: <em> list of lists </em> \n
            The coorinate pairs for each node.
        @param optimum: \e float \n
            The optimal solution. \n
        """

        ## @var name
        # \e string: The name of the TSPLIB problem.
        self.name = name

        ## @var dimension
        # \e integer: The number of nodes (cities) in the problem.
        self.dimension = dimension

        ## @var nodes
        # <em> list of lists: </em> The coorinate pairs for each node.
        self.nodes = nodes

        ## @var optimum
        # \e float:  The optimal solution.
        self.optimum = optimum

    def __repr__(self):
        """!
        TSP class param print function.

        @param self: \e TSP pointer \n
            The TSP pointer. \n
        """
        return "TSP Definition({}, {}, {})".format(self.name, self.dimension,
                                                   self.nodes)

    def __str__(self):
        """!
        Human readable TSP print function.

        @param self: \e TSP pointer \n
            The TSP pointer. \n
        """

        header = ["  TSP Definition:"]
        header += ["TSPLIB Name: {}".format(self.name)]
        header += ["# of Cities: {}".format(self.dimension)]
        header += ["Locations:"]
        header = "\n".join(header)+"\n"
        header += "\n".join(["{0:<7}{1}".format(x, y) for x, y in self.nodes])
        return header

    def read_tsp(self, filename):
        """!
        Read the starting TSP points from a TSPLIB standard file and populate
        class attributes.

        @param filename: \e string \n
            Path and filename of the tsp problem. \n
        """

        # Initialize variables
        header = True
        self.nodes = []

        # Open file, read line by line, and store tsp problem parameters to
        # the TSP object
        with open(filename, 'r') as f:
            line = f.readline()
            key, value = line.split(":")
            self.name = value.strip()
            for line in f:
                if header == True:
                    key, value = line.split(":")
                    if key.strip() == 'DIMENSION':
                        self.dimension = int(value.strip())
                    if key.strip() == 'OPTIMUM':
                        self.optimum = float(value.strip())
                    if key.strip() == 'EDGE_WEIGHT_TYPE':
                        header = False
                        next(f)
                elif header == False:
                    if line.strip() != 'EOF':
                        splitList = line.split()
                        if len(splitList) != 3:
                            raise ValueError('Line {}: {} has {} spaces, '
                                             'expected 1.'.format(line,
                                             line.rstrip(),
                                             len(splitList)-1))
                        else:
                            self.nodes.append([float(splitList[1].strip()),
                                               float(splitList[2].strip())])

        # Test that the file closed
        assert f.closed == True, "File did not close properly."

    def build_prob_params(self, probParams):
        """!
        Takes the current class attributes and populates a ProblemParameters
        object for use in optimization algorithms.

        @param probParams <em> ProblemParameters object </em> \n
            A problem parameters object to be initialized with the class
            parameters. \n
        """

        probParams.objective = ObjectiveFunction('tsp')
        probParams.lb = np.zeros(self.dimension)
        probParams.ub = np.ones_like(probParams.lb)*(self.dimension-1)
        probParams.varType = ['x' for v in probParams.ub]
        probParams.discreteVals = [self.nodes] * self.dimension
        probParams.optimum = self.optimum
        probParams.pltTitle = self.name
        probParams.histTitle = self.name
        probParams.cID = np.zeros(self.dimension)
        probParams.iID = np.zeros(self.dimension)
        probParams.dID = np.zeros(self.dimension)
        probParams.xID = np.ones(self.dimension)
        