"""!
@file Benchmarks/HyperOpt.py
@package Gnowee

@defgroup HyperOpt HyperOpt

@brief Performs a hyperoptimization on the Gnowee parameters

Used to provide optimization of parameters used in Gnowee.  As of this version,
it separately considers the TSP problems.  Saves and plots the results.

For examples on how to run HyperOpt, please refer to the ContinuousHyperOpt or
TSPHyperOpt notebooks included in the Benchmarks directory.

@author James Bevins

@date 19May17
"""

import sys
import os
sys.path.insert(0,os.path.pardir+'/src')
import Gnowee

import numpy as np
import OptiPlot as op
import math as m
import copy as cp

from operator import attrgetter
from collections import Counter
from scipy.stats import rankdata
from ObjectiveFunction import ObjectiveFunction
from GnoweeUtilities import ProblemParameters, Event
from GnoweeHeuristics import GnoweeHeuristics
from TSP import TSP

#------------------------------------------------------------------------------#
def hyper_opt(tspOn, param, paramList, numIter=25):
    """!
    Performs a hyper optimization on the Gnowee control paramenters for both
    TSP and standard mixed integer, condinuous, and discrete problems.

    The parameters considered are hardwired.  It is kind of crappy and only
    allows one parameter to be optimized at a time.

    @param tspOn: \e boolean \n 
        If true, the settings will be optimized for TSP problems. \n
    @param param: \e string \n 
        An attribute name of the GnoweeHeuristics object to be optimized. \n
    @param paramList: \e list \n 
        List of the parameters to be considered in the optimization. \n
    @param numIter: \e integer \n 
        The number of iterations to perform for each parameter. \n
   
    @return <em> numpy array: </em> Returns the results of the hyper-
        optimization. \n
    """

    assert tspOn == False or tspOn == True, 'tspOn must be True or False.'

    # Build list of optimization problem types 
    if tspOn == False:
        optFunct = ['welded_beam', 'pressure_vessel', 'speed_reducer', 'spring',
                  'mi_spring', 'mi_pressure_vessel', 'mi_chemical_process',
                  'ackley', 'dejong', 'easom', 'griewank', 'rastrigin',
                  'rosenbrock']
        numProb = len(optFunct)
        dimension = [0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 6, 5, 5]

    else:
        optFunct = ['eil51.tsp', 'st70.tsp', 'pr107.tsp', 'bier127.tsp',
                    'ch150.tsp']
        tspPath = [os.path.pardir+'/Benchmarks/TSPLIB/eil51.tsp',
                   os.path.pardir+'/Benchmarks/TSPLIB/st70.tsp',
                   os.path.pardir+'/Benchmarks/TSPLIB/pr107.tsp',
                   os.path.pardir+'/Benchmarks/TSPLIB/bier127.tsp',
                   os.path.pardir+'/Benchmarks/TSPLIB/ch150.tsp']
        numProb = len(tspPath)
        dimension = [0, 0, 0, 0, 0]

    # Initialize variables
    maxIter = numIter  #Number of algorithm iterations
    results = np.zeros((numProb, len(paramList), 3))

    for i in range(0, numProb, 1):
        for j in range(0, len(paramList), 1):
            history=[]   # Contains the final timeline results from each run 
            
            # Set default optimization settings
            gh = GnoweeHeuristics()
            if tspOn == False:
                gh.set_preset_params(optFunct[i], 'Gnowee',
                                 dimension=dimension[i])
            else:
                gh.stallLimit = 15000
                tspProb = TSP()
                tspProb.read_tsp(tspPath[i])
                tspProb.build_prob_params(gh)

            # Update current settings
            setattr(gh, param, paramList[j])
            
            for k in range(0, maxIter, 1):
                print "Run: {}, {}={}, iter# {}".format(optFunct[i], param,
                                                   paramList[j], k)

                #Run Optimization Algorithm
                (timeline) = Gnowee.main(gh)

                # Save final timeline data for future processing
                minGen = min(timeline, key=attrgetter('fitness'))
                history.append(Event(minGen.generation, minGen.evaluations, 
                                     minGen.fitness, minGen.design))

            # Calculate averages and standard deviations
            tmp = []
            averages = Event(sum(c.generation for c in history)\
                             /float(len(history)),
                             sum(c.evaluations for c in history)\
                             /float(len(history)),
                             sum(c.fitness for c in history)\
                             /float(len(history)),tmp)

            if maxIter > 1:
                for l in range(len(history[-1].design)):
                    stdDev = Event(m.sqrt(sum([(c.generation \
                                - averages.generation)**2 for c in history])\
                                /(len(history) - 1)),
                              m.sqrt(sum([(c.evaluations \
                                - averages.evaluations)**2 for c in history])\
                                /(len(history) - 1)),
                              m.sqrt(sum([(c.fitness - averages.fitness)**2 \
                                for c in history])/(len(history) - 1)), tmp)
            else:
                tmp = np.zeros(len(history[-1].design))
                stdDev = Event(0, 0, 0.0, tmp)

            # Save Results for plotting
            results[i, j, 0] = paramList[j]
            if gh.optimum == 0.0:
                results[i, j, 1] = (averages.fitness*(averages.evaluations+3\
                                                      *stdDev.evaluations))
            else:
                results[i ,j , 1] = (abs((averages.fitness-gh.optimum)\
                                         /gh.optimum)*(averages.evaluations+3\
                                                       *stdDev.evaluations))
            if gh.optimum !=0:
                results[i, j, 2] = m.sqrt((stdDev.fitness/averages.fitness)**2\
                                          +(stdDev.evaluations\
                                            /averages.evaluations)**2)\
                                          /gh.optimum*results[1, j, 1]
            else:
                results[i, j, 2]=m.sqrt((stdDev.fitness/averages.fitness)**2\
                                        +(stdDev.evaluations\
                                          /averages.evaluations)**2)\
                                        *results[1,j,1]

    # Output the results and some statistics 
    print repr(results)
    print ("Variable   Weighted Relative Sum (Low is Better)  "
           "Sum of Ordinal Rank   Number of Minimums")
    print ("==================================================="
          "=======================================")
    weighted = cp.copy(results)
    for i in range(0,len(results[0])):
        minLoc = []
        ranks = []
        for j in range(0,len(results)):
            weighted[j,:,1]=results[j,:,1]/min(results[j,:,1])
            minLoc.append(np.argmin(results[j,:,1]))
            ranks.append(rankdata(results[j,:,1], method='ordinal'))
        print ("{}                  {}                          "
               "{}                  {}".format(paramList[i], sum(weighted[:,i,1]),
                                               sum(np.asarray(ranks)[:,i]), Counter(minLoc)[i]))

    #Plot the results
    if tspOn == False:
        label = ['\\textbf{Welded Beam}', '\\textbf{Pressure Vessel}',
                 '\\textbf{Speed Reducer}', '\\textbf{Spring}',
                 '\\textbf{MI Spring}', '\\textbf{MI Pressure Vessel}',
                 '\\textbf{MI Chemical Process}']
        title = '\\textbf{Hyper-Optimization of Gnowee Algorithm for %s}' %param
        op.plot_optimization(results[0:len(label)], label, title)
        label2 = ['\\textbf{Ackley}', '\\textbf{De Jong}', '\\textbf{Easom}',
                 '\\textbf{Griewank}', '\\textbf{Rastrigin}',
                 '\\textbf{Rosenbrock}']
        op.plot_optimization(results[len(label):len(label)+len(label2)],
                                  label2, title)
    else:
        label = ['\\textbf{Eil51}', '\\textbf{St70}', '\\textbf{Pr107}',
                 '\\textbf{Bier127}', '\\textbf{Ch150}']
        title = '\\textbf{Hyper-Optimization of Gnowee Algorithm for %s}' %param
        op.plot_optimization(results, label, title)
    
    return results 
    