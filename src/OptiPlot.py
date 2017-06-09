"""!
@file src/OptiPlot.py
@package Gnowee

@defgroup OptiPlot OptiPlot

@brief Plotting functions developed to help visualize and quantify the
metaheuristic optimization process.

@author James Bevins

@date 5Jun17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter
from Sampling import levy
from scipy import integrate
from math import ceil, pi, exp, cos

#------------------------------------------------------------------------------#
def plot_vars(data, lowBounds=[], upBounds=[], title=[], label=[]):
    """!
    @ingroup OptiPlot
    Plot the variables as they change in the optimization process.  Currently
    only functions in post-processing, not real time.

    @param data: <em> list of event objects </em> \n
        Contain the optimization history in event objects within the data
        list. \n
    @param lowBounds: \e array \n
        The lower bounds of the design variable(s). \n
    @param upBounds: \e array \n
        The upper bounds of the design variable(s). \n
    @param title: \e string \n
        Title for plot. \n
    @param label: list \n
        List of names of design variables. \n
    """

    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    plt.rcParams['savefig.dpi'] = 900
    majorFormatter = FormatStrFormatter('%0.1e')

    # Establish labels for each data set and title for the plot
    if label == []:
        label.append('\\textbf{Fitness}')
        for i in range(2, len(data[0].design), 1):
            label.append('\\textbf{Var\#}' + str(i-1))
    if title == []:
        title = "Optimization Results for Each Design Variable"

    # Build 1st Subplot - Fitness plot
    fig = plt.figure()
    ax = fig.add_subplot(len(data[0].design)+1, 1, 1)
    ax.set_title(title, y=1.08)
    x = [tmp.generation for tmp in data]
    y = [tmp.fitness for tmp in data]
    ax.plot(x, y, 'ko-')
    ax.set_ylabel(label[0], fontsize=15, x=-0.04)

    # Format fitness plot
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.set_major_formatter(majorFormatter)
    #if all(y) > 0:
    #    ax.set_yscale('log')
    ax.set_ylim(min(y), data[2].fitness)

    # Add text stating fmin to plot
    ax.text(0.8, 0.8, '\\textbf{fmin = %f}' %data[-1].fitness, ha='center',
            va='center', transform=ax.transAxes)

    # Add in subplots for each design variable
    for i in range(0, len(data[0].design), 1):
        ax = fig.add_subplot(len(data[0].design)+1, 1, i+2)
        y = [tmp.design[i] for tmp in data]
        ax.plot(x, y, 'ko-')
        ax.set_ylabel(label[i+1], fontsize=15, x=-0.04)

        # Format subplot
        ax.axes.get_xaxis().set_visible(False)
        ax.yaxis.set_major_formatter(majorFormatter)
        if len(lowBounds) != 0:
            assert len(upBounds) == len(lowBounds), ('Boundaries have '
                      'different # of design variables in plot_vars function.')
            assert (len(data[0].design)) == len(lowBounds), ('Data has '
                   'different # of design variables than bounds in plot_vars.')
            ax.set_ylim([lowBounds[i], upBounds[i]])
            ax.set_yticks(np.arange(lowBounds[i], upBounds[i]+0.01,
                                    0.25*(upBounds[i]-lowBounds[i])))

        # Add text stating final design value to plot
        ax.text(0.8, 0.8, '\\textbf{Optimum %s = %f}' %(label[i+1],
                                                        data[-1].design[i]),
                ha='center', va='center', transform=ax.transAxes)

    # Turn on X axis below final subplot
    ax.axes.get_xaxis().set_visible(True)
    ax.set_xlabel('\\textbf{Generation}', fontsize=15, y=-0.04)
    #plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.show()

#------------------------------------------------------------------------------#
def plot_hist(data, title='', xLabel=''):
    """!
    @ingroup OptiPlot
    Plots the histogram of function evaluation results from multiple runs of
    an optimization algorithm.  Can be used to understand the convergence of
    the algorithm.

    @param data: \e list \n
        Contains the number of function evals for each optimization run.
    @param title: \e string \n
        Title for plot. \n
    @param xLabel: \e string \n
        Label for independent variable. \n
    """

    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    plt.rcParams['savefig.dpi'] = 900
    majorFormatter = FormatStrFormatter('%0.1e')

    # Establish labels for each data set and title for the plot
    if xLabel == '':
        xLabel = ('\\textbf{# Function Evals}')
    yLabel = ('\\textbf{Probability}')
    if title == '':
        title = "Histogram of Function Evaluations for Optimization"

    # Plot Histogram
    num = len(data)
    w = np.ones_like(data)/float(num)
    plt.hist(data, bins=100, weights=w, facecolor='black')

    # Plot Labels
    plt.xlabel('\\textbf{%s}' %xLabel, fontsize=15, y=-01.04)
    plt.ylabel('\\textbf{%s}' %yLabel, fontsize=15, x=-0.04)
    plt.title('\\textbf{%s}' %title, fontsize=17, y=1.04)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    # Tweak spacing to prevent clipping of yLabel
    plt.subplots_adjust(left=0.15)

    plt.show()

#------------------------------------------------------------------------------#
def plot_hist_comp(data, data2, dataLabels, title='', xLabel=''):
    """!
    @ingroup OptiPlot
    Histograms and plots the comparison of two sets of function evaluation data.

    @param data: \e list \n
        Contains the number of function evals for each optimization run. \n
    @param data2: \e list \n
        Contains the number of function evals for each optimization run for a
        second set of runs. \n
    @param dataLabels: \e list \n
        Contains the legend label names for each data set. \n

    @param title: \e string \n
        Title for plot. \n
    @param xLabel: \e string \n
        Label for independent variable. \n
    """

    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    plt.rcParams['savefig.dpi'] = 900

    # Establish labels for each data set and title for the plot
    if xLabel == '':
        xLabel = ('\\textbf{# Function Evals}')
    yLabel = ('\\textbf{Probability}')
    if title == '':
        title = "Histogram of Function Evaluations for Optimization"

    # Plot Histogram
    num = len(data)
    w = np.ones_like(data)/float(num)

    maxVal = max([max(data), max(data2)])
    bins = np.arange(0, maxVal, ceil(maxVal/100))

    plt.hist(data, bins=bins, weights=w, facecolor='black',
             histtype='stepfilled', alpha=1.0, label=dataLabels[0])
    plt.hist(data2, bins=bins, weights=w, facecolor='grey',
             histtype='stepfilled', alpha=0.85, label=dataLabels[1])

    # Plot Labels
    plt.xLabel('\\textbf{%s}' %xLabel, fontsize=15, y=-01.04)
    plt.yLabel('\\textbf{%s}' %yLabel, fontsize=15, x=-0.04)
    plt.title('\\textbf{%s}' %title, fontsize=17, y=1.04)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    # Tweak spacing to prevent clipping of yLabel
    plt.subplots_adjust(left=0.15)

    # Add legend
    plt.legend(borderaxespad=0.75, loc=1, fontsize=14, handlelength=5,
               borderpad=0.5, labelspacing=0.75, fancybox=True, framealpha=0.5)
    plt.show()

#------------------------------------------------------------------------------#
def plot_feval_hist(data=[], listData=[], label=[]):
    """!
    @ingroup OptiPlot
    Plots the fitness vs function evaluation results of an optimization
    algorithm run.  Can plot a single run or multiple to compare results. To
    plot multiple data sets, use the listData argument; otherwise, use the
    data argument.

    @param data: <em> list or array </em> \n
        Contains the function eval history. Columns are: [function evals,
        fitness, number of datapoints]. \n
    @param listData: <em> list of lists or arrays </em> \n
        Contains a list of function eval histories. Columns are:
        [function evals, fitness, number of datapoints]. \n
    @param label: \e list \n
        List of names corresponding to the data sets provided. \n
    """

    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    plt.rcParams['savefig.dpi'] = 900
    majorFormatter = FormatStrFormatter('%0.1e')

    # Label and markers
    marker = ['ko-', 'k^-', 'k+-', 'ks-', 'kd-', 'k*-', 'k>-']
    if label == []:
        label = ['\\textbf{GA}', '\\textbf{SA}', '\\textbf{PSO}',
                 '\\textbf{CS}', '\\textbf{MCS}', '\\textbf{DMC}']

    # Build Plot if only one set of data passed
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if len(data) != 0:
        x = data[:, 0]
        y = data[:, 1]
        ax.plot(x, y, 'ko-')
        #yerr = data[:, 2]
        #ax.errorbar(x, y, yerr=yerr, fmt='ko-')
    elif len(listData) != 0:
        for i in range(0, len(listData), 1):
            x = listData[i][:, 0]
            y = listData[i][:, 1]
            ax.plot(x, y, marker[i], label=label[i])
            #yerr = listData[i][:, 2]
            #ax.errorbar(x, y, yerr=yerr, fmt=marker[i],label=label[i])

            # Add and locate legend
            plt.legend(borderaxespad=0.75, loc=1, fontsize=14, handlelength=5,
                       borderpad=0.5, labelspacing=0.75, fancybox=True,
                       framealpha=0.5)

    # Format plot
    ax.set_title('\\textbf{Average Deviation from Optimal Fitness}',
                 fontsize=18, y=1.04)
    ax.set_ylabel('\\textbf{\% Difference from Optimal Fitness}',
                  fontsize=18, x=-0.04)
    ax.yaxis.set_major_formatter(majorFormatter)
    if all(y) > 0:
        ax.set_yscale('log')
    ax.set_ylim(np.min(y), y[1])
    ax.set_xlabel('\\textbf{Function Evaluations}', fontsize=18, y=-0.04)
    #ax.set_xscale('log')
    ax.set_xlim(x[1], np.max(x))

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.show()

#------------------------------------------------------------------------------#
def plot_tlf(alpha=1.5, gamma=1., numSamp=1E7, cutPoint=10.):
    """!
    @ingroup OptiPlot
    Plots a comparison of the TLF to the Levy distribution.

    @param alpha: \e float \n
        Levy exponent - defines the index of the distribution and controls
        scale properties of the stochastic process.
    @param gamma: \e float \n
        Gamma - Scale unit of process for Levy flights. \n
    @param numSamp: \e integer \n
        Number of Levy flights to sample. \n
    @param cutPoint: \e float \n
        Point at which to cut sampled Levy values and resample. \n
    """

    # Initialize variables
    l = []  #List to store Levy distribution values
    bins = np.array(range(0, int(cutPoint+1), 1))/cutPoint

    # Calculate the Levy distribution
    for i in range(0, len(bins)):
        l.append(1/pi*integrate.quad(lambda x: exp(-gamma*x**(alpha)) \
                                     *cos(x*bins[i]*cutPoint),
                                     0, float("inf"), epsabs=0, epsrel=1.e-5,
                                     limit=150)[0]*2)

    # Draw numSamp samples from the Levy distribution
    levySamp = abs(levy(1, numSamp)/cutPoint).reshape(numSamp)

    # Resample values above the range (0,1)
    for i in range(len(levySamp)):
        while levySamp[i] > 1:
            levySamp[i] = abs(levySamp(1, 1)/cutPoint).reshape(1)

    #Plot the TLF and Levy distribution on the interval (0,1)
    w = np.ones_like(levySamp)/float(numSamp)   #Weights to normalize histogram
    plt.rc('text', usetex=True)
    ax = plt.subplot(111)
    ax.hist(levy, bins=bins, facecolor='grey', weights=w)
    ax.plot(bins, l, color='k', linewidth=3.0)
    ax.set_yscale("log")
    plt.xLabel('\\textbf{z}', fontsize=15, y=-0.04)
    plt.yLabel('\\textbf{P(z)}', fontsize=15, x=-0.04)
    plt.title('\\textbf{Comparison of TLF and Levy Function}', fontsize=17)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.show()

#------------------------------------------------------------------------------#
def plot_optimization(data, label, title='', xLabel=''):
    """!
    @ingroup OptiPlot
    Plots the results of optimization process for a given algorithm and
    parameter.

    @param data: \e array \n
        Contains the function eval history. Columns are: [function evals,
        fitness, number of datapoints]
    @param label: \e list \n
        List of names of the problem types ran. \n
    @param title: \e string \n
        Title for plot. \n
    @param xLabel: \e string \n
        Title for x-axis. \n
    """

    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    plt.rc('axes', linewidth=1.5)
    plt.rc('font', weight='bold')
    plt.rcParams['savefig.dpi'] = 900
    majorFormatter = FormatStrFormatter('%0.1e')

    # Markers; currently hard wired
    marker = ['ko-', 'k^-', 'k+-', 'ks-', 'kd-', 'k*-', 'k>-']

    # Build Plot if only one set of data passed
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(0, len(data), 1):
        x = data[i, :, 0]
        y = data[i, :, 1]
        ax.plot(x, y, marker[i], label=label[i])
        #yerr = data[i, :, 2]
        #ax.errorbar(x, y, yerr=yerr, fmt=marker[i], label=label[i])

        # Add and locate legend
        plt.legend(borderaxespad=0.75, loc=1, fontsize=12, handlelength=5,
                   borderpad=0.5, labelspacing=0.75, fancybox=True,
                   framealpha=0.5)

    # Format plot
    if title == '':
        ax.set_title('\\textbf{Optimization Results}',
                     fontsize=17, y=1.04)
    else:
        ax.set_title(title, y=1.04)
    ax.set_ylabel('\\textbf{Performance Metric}', fontsize=16, x=-0.04)
    ax.yaxis.set_major_formatter(majorFormatter)
    if all(y) > 0:
        ax.set_yscale('log')
    ax.set_ylim(0.8*np.min(data[:, :, 1]), 1.2*np.max(data[:, :, 1]))
    if xLabel == '':
        ax.set_xlabel('\\textbf{Parameter Value}', fontsize=16, y=-0.04)
    else:
        ax.set_xlabel(xLabel, fontsize=16, y=-0.04)    
    #ax.set_xscale('log')
    ax.set_xlim(x[0], np.max(x))

    ax.xaxis.set_tick_params(which='major', width=2, labelsize=16, length=5)
    ax.yaxis.set_tick_params(which='major', width=2, labelsize=16, length=5)
    ax.xaxis.set_tick_params(which='minor', width=1.5, length=4)
    ax.yaxis.set_tick_params(which='minor', width=1.5, length=4)

    plt.show()
