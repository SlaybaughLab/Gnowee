#######################################################################################################
#
# Module : OptiPlot.py
#
# Contains : Plotting functions developed to help visualize and quantify the metaheuristic optimization 
#            process
#
# Author : James Bevins
#
# Last Modified: 8Oct15
#
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mpt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import SamplingMethods as sm
from scipy import integrate
import math as m
import copy as cp

#---------------------------------------------------------------------------------------#  
def Plot_Vars(data,low_bounds=[],up_bounds=[],title=[],label=[],debug=False):
    """
    Perform a cuckoo search optimization (CS)
   
    Parameters
    ==========
    data : list of event objects
        Contain the optimization history in event objects within the data list
        Attributes are: generation (.g), function evaluations (.e), fitness (.f), and design (.d)
    Optional
    ========   
    low_bounds : array
        The lower bounds of the design variable(s)
    up_bounds : array
        The upper bounds of the design variable(s)
    title : string
        Title for plot
        (Default: [])
    label : list
        List of names of design variables
        (Default: [])
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    None, generates plot of design variables vs generation
   
    """

    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    majorFormatter = FormatStrFormatter('%0.1e')

    # Establish labels for each data set and title for the plot
    if label==[]:
        label.append('\\textbf{Fitness}')
        for i in range(2,len(data[0].d),1):
            label.append('\\textbf{Var\#}' + str(i-1))
    if title==[]:
        title="Optimization Results for Each Design Variable"
    
    # Build 1st Subplot - Fitness plot
    fig=plt.figure()
    ax=fig.add_subplot(len(data[0].d)+1,1,1)
    ax.set_title(title, y=1.08)
    x=[tmp.g for tmp in data]
    y=[tmp.f for tmp in data]
    ax.plot(x,y, 'ko-')
    ax.set_ylabel(label[0],fontsize=15, x=-0.04) 
    
    # Format fitness plot
    ax.axes.get_xaxis().set_visible(False)  
    ax.yaxis.set_major_formatter(majorFormatter)
#    if all(y) > 0:
#        ax.set_yscale('log')
    ax.set_ylim(min(y),data[2].f)
    

    # Add text stating fmin to plot
    ax.text(0.8, 0.8,'\\textbf{fmin = %f}' %data[-1].f, ha='center', va='center', transform=ax.transAxes)
    
    # Add in subplots for each design variable
    for i in range(0,len(data[0].d),1):
        ax=fig.add_subplot(len(data[0].d)+1,1,i+2)
        y=[tmp.d[i] for tmp in data]
        ax.plot(x, y, 'ko-')
        ax.set_ylabel(label[i+1],fontsize=15, x=-0.04)
        
        # Format subplot
        ax.axes.get_xaxis().set_visible(False)
        ax.yaxis.set_major_formatter(majorFormatter)
        if low_bounds != []:
            assert len(up_bounds)==len(low_bounds), 'Boundaries have different # of design variables in Plot_Vars function.'
            assert (len(data[0].d))==len(low_bounds), 'Data has different # of design variables than bounds in Plot_Vars.'
            ax.set_ylim([low_bounds[i],up_bounds[i]])
            ax.set_yticks(np.arange(low_bounds[i],up_bounds[i]+0.01, 0.25*(up_bounds[i]-low_bounds[i])))
            
        # Add text stating final design value to plot
        ax.text(0.8, 0.8,'\\textbf{Optimum %s = %f}' %(label[i+1],data[-1].d[i]), ha='center', 
                va='center', transform=ax.transAxes)

    # Turn on X axis below final subplot
    ax.axes.get_xaxis().set_visible(True)
    ax.set_xlabel('\\textbf{Generation}',fontsize=15, y=-0.04)
    #plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)  
    
    plt.show()
    
#---------------------------------------------------------------------------------------#       
def Plot_Hist(data,title=[],xlabel='',ylabel=[],debug=False):
    """
    Perform a cuckoo search optimization (CS)
   
    Parameters
    ==========
    data : list 
        Contains the number of function evals for each optimization run
   
    Optional
    ========   
    title : string
        Title for plot
        (Default: [])
    xlabel : string
        Label for independent variable
        (Default: [])
    ylabel : list
        List of names of design variables
        (Default: [])
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    None, generates plot of design variables vs generation
   
    """
        
    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    majorFormatter = FormatStrFormatter('%0.1e')

    # Establish labels for each data set and title for the plot
    if xlabel=='':
        xlabel=('\\textbf{# Function Evals}')
    if ylabel==[]:
        ylabel=('\\textbf{Probability}')
    if title==[]:
        title="Histogram of Function Evaluations for Optimization"

    # Plot Histogram
    num=len(data)
    w=np.ones_like(data)/float(num)
    plt.hist(data, bins=100, weights=w, facecolor='black')

    # Plot Labels
    plt.xlabel('\\textbf{%s}' %xlabel,fontsize=15, y=-01.04)
    plt.ylabel('\\textbf{%s}' %ylabel,fontsize=15, x=-0.04)
    plt.title('\\textbf{%s}' %title,fontsize=17, y=1.04)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)  
    
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    
    plt.show()
    
#---------------------------------------------------------------------------------------#       
def Plot_Hist_Comp(data,data2,data_labels,title='',xlabel='',ylabel=[],debug=False):
    """
    Histograms and plots the comparison of two sets of function evaluation data
   
    Parameters
    ==========
    data : list 
        Contains the number of function evals for each optimization run
    data2 : list 
        Contains the number of function evals for each optimization run
    data_labels : list 
        Contains the legend label names for each data set
   
    Optional
    ========   
    title : string
        Title for plot
        (Default: '')
    xlabel : string
        Label for independent variable
        (Default: [])
    ylabel : list
        List of names of design variables
        (Default: [])
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    None, generates plot of design variables vs generation
   
    """
    
    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    majorFormatter = FormatStrFormatter('%0.1e')

    # Establish labels for each data set and title for the plot
    if xlabel=='':
        xlabel=('\\textbf{# Function Evals}')
    if ylabel==[]:
        ylabel=('\\textbf{Probability}')
    if title=='':
        title="Histogram of Function Evaluations for Optimization"

    # Plot Histogram
    iter=len(data)
    w=np.ones_like(data)/float(iter)

    m_val=max([max(fevals),max(fevals_sa)])
    bins=np.arange(0,m_val,m.ceil(m_val/100))

    plt.hist(data, bins=bins, weights=w, facecolor='black', histtype='stepfilled', alpha=1.0, label=data_label[0])
    plt.hist(data2, bins=bins, weights=w, facecolor='grey', histtype='stepfilled', alpha=0.85, label=data_label[1])

    # Plot Labels
    plt.xlabel('\\textbf{%s}' %xlabel,fontsize=15, y=-01.04)
    plt.ylabel('\\textbf{%s}' %ylabel,fontsize=15, x=-0.04)
    plt.title('\\textbf{%s}' %title,fontsize=17, y=1.04)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)  

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)

    plt.legend(borderaxespad=0.75, loc=1, fontsize=14, 
                            handlelength=5, borderpad=0.5, labelspacing=0.75, fancybox=True, 
                            framealpha=0.5)

    plt.show()

#---------------------------------------------------------------------------------------#   
def Plot_Feval_Hist(data=[],listdata=[],label=[],debug=False):
    """
    Plots the fitness vs function evaluation results of an optimization algorithm run.  
    Can plot a single run or multiple to compare results.  
   
    Parameters
    ==========
    Optional
    ========  
    data : array
        Contain the function eval history 
        Columns are: [function evals, fitness, number of datapoints]
        (Default: Null)
    listdata : list
        Contains a list of arrays containing the function eval history 
        Columns are: [function evals, fitness, number of datapoints]
        (Default: Null)
    label : list
        List of names of the optimization types ran
        (Default: Null)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    None, generates plot of design variables vs generation
   
    """
        
    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    majorFormatter = FormatStrFormatter('%0.1e')
    
    # Label and markers
    marker=['ko-','k^-','k+-','ks-','kd-','k*-',]
    if label==[]:
        label=['\\textbf{GA}','\\textbf{SA}','\\textbf{PSO}','\\textbf{CS}','\\textbf{MCS}','\\textbf{DMC}']
    
    # Build Plot if only one set of data passed
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    if data!=[]:
        x=data[:,0]
        y=data[:,1]
        yerr=data[:,2]
        ax.plot(x,y, 'ko-')
#        ax.errorbar(x, y, yerr=yerr, fmt='ko-')
    elif listdata != []:
        for i in range(0,len(listdata),1):
            x=listdata[i][:,0]
            y=listdata[i][:,1]
            yerr=listdata[i][:,2]
            ax.plot(x,y,marker[i],label=label[i]) 
#            ax.errorbar(x, y, yerr=yerr, fmt=marker[i],label=label[i])   
            # Add and locate legend
            leg = ax.legend()
            plt.legend(borderaxespad=0.75, loc=1, fontsize=14, 
                        handlelength=5, borderpad=0.5, labelspacing=0.75, fancybox=True, 
                        framealpha=0.5);
            
    # Format plot
    ax.set_title('\\textbf{Average Deviation from Optimal Fitness}',fontsize=18, y=1.04)
    ax.set_ylabel('\\textbf{\% Difference from Optimal Fitness}',fontsize=18, x=-0.04) 
    ax.yaxis.set_major_formatter(majorFormatter)
    if all(y)>0:
        ax.set_yscale('log')
    ax.set_ylim(np.min(y),y[1])
    ax.set_xlabel('\\textbf{Function Evaluations}',fontsize=18, y=-0.04) 
    #ax.set_xscale('log')
    ax.set_xlim(x[1],np.max(x))
      
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)  

    
    plt.show()
    
#---------------------------------------------------------------------------------------#
def Plot_TLF(alpha=1.5,gamma=1.,num_samp=1E7,cut_point=10.,debug=False):
    
    """
    Plots the comparison of the TLF to the Levy distribution
   
    Parameters
    ==========
  
    Optional
    ========   
    alpha : scalar
        Levy exponent - defines the index of the distribution and controls scale properties of the stochastic process
        (Default: 1.5)
    gamma : scalar
        Gamma - Scale unit of process for Levy flights (Default: 1.)
    num_samp : integer
        Number of Levy flights to sample (Default: 1E7)
    cut_point : scalar
        Point at which to cut sampled Levy values and resample
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    None, generates plot
    """
    
    # Initialize variables
    l=[]  #List to store Levy distribution values
    bins=np.array(range(0,int(cut_point+1),1))/cut_point  #Bins for histogram and Levy calculation

    # Calculate the Levy distribution
    for i in range(len(bins)):
        l.append(1/m.pi*integrate.quad(lambda x: m.exp(-gamma*x**(alpha))*m.cos(x*bins[i]*cut_point), 
                                       0, float("inf"),epsabs=0, epsrel=1.e-5,limit=150)[0]*2)
    
    # Draw num_samp samples from the Levy distribution
    levy=abs(sm.Levy(1,num_samp)/cut_point).reshape(num_samp)
    
    if debug==True:
        print "Prior to resampling, the maximum sampled value is:", np.max(levy)
        print "There are ",(levy<0.2).sum(), " < 0.2"
        print (levy>1.0).sum()/num_samp,"% of the samples are above the cut point."
    
    # Resample values above the range (0,1)
    for i in range(len(levy)):
        while levy[i]>1:
            levy[i]=abs(sm.Levy(1,1)/cut_point).reshape(1)

    #Plot the TLF and Levy distribution on the interval (0,1)
    w=np.ones_like(levy)/float(num_samp)   #Weights to normalize histogram
    plt.rc('text', usetex=True)
    ax=plt.subplot(111)
    ax.hist(levy, bins=bins, facecolor='grey', weights=w)
    ax.plot(bins,l,color='k',linewidth=3.0,)
    ax.set_yscale("log")
    plt.xlabel('\\textbf{z}',fontsize=15, y=-0.04)
    plt.ylabel('\\textbf{P(z)}',fontsize=15, x=-0.04)
    plt.title('\\textbf{Comparison of TLF and Levy Function}',fontsize=17)
    
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    
    plt.show()    
    
#---------------------------------------------------------------------------------------#   
def Plot_Meta_Optimization(data,label,title='',debug=False):
    """
    Plots the results of meta-optimization process for a given algorithm and parameter.  
   
    Parameters
    ==========
    data : array
        Contain the function eval history 
        Columns are: [function evals, fitness, number of datapoints]
    label : list
        List of names of the problem types ran
    Optional
    ========  
    title : string
        Title for plot
        (Default: '')
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    None, generates plot of design variables vs generation
   
    """
        
    # Allow use of Tex sybols and set formats
    plt.rc('text', usetex=True)
    majorFormatter = FormatStrFormatter('%0.1e')
    
    # Markers; currently hard wired
    marker=['ko-','k^-','k+-','ks-','kd-','k*-']
    
    # Build Plot if only one set of data passed
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for i in range(0,len(data),1):
        x=data[i,:,0]
        y=data[i,:,1]
        yerr=data[i,:,2]
        ax.plot(x,y,marker[i],label=label[i]) 
#        ax.errorbar(x, y, yerr=yerr, fmt=marker[i],label=label[i])   
        # Add and locate legend
        leg = ax.legend()
        plt.legend(borderaxespad=0.75, loc=1, fontsize=12, 
                   handlelength=5, borderpad=0.5, labelspacing=0.75, fancybox=True, 
                   framealpha=0.5);
            
    # Format plot
    if title=='':
        ax.set_title('\\textbf{Meta-Optimization of DMC Algorithm}',fontsize=17, y=1.04)
    else:
        ax.set_title(title, y=1.04)
    ax.set_ylabel('\\textbf{Performance Metric}',fontsize=15, x=-0.04) 
    ax.yaxis.set_major_formatter(majorFormatter)
    if all(y)>0:
        ax.set_yscale('log')
    ax.set_ylim(np.min(y),y[1])
    ax.set_xlabel('\\textbf{Parameter Value}',fontsize=15,y=-0.04) 
    #ax.set_xscale('log')
    ax.set_xlim(x[1],np.max(x))
    
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    
    plt.show()
