#######################################################################################################
#
# Module : HyperOpt.py
#
# Contains : Used to provide optimization on parameters used in Gnowee.  As of this version, it separately considers
#            the discrete and continuous.  Saves and plots the results.
#
# Author : James Bevins
#
# Last Modified: 15Aug16
#
###############################################################################################################################

import sys
import os
sys.path.insert(0,os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir)))

import numpy as np
import ObjectiveFunctions as of
import OptiPlot as op
import ContinuousGnowee as cg
import DiscreteGnowee as dg
import Utilities as util
import math as m
from operator import attrgetter

#---------------------------------------------------------------------------------------#
def Hyper_Opt(Continuous_on,Discrete_on):
    """
    Performs a hyper optimization on the Gnowee control paramenters for both discrete and 
    continuous versions.  Currently the parameter to be optimized is hard wired.
    
    Parameters
    ==========
    Continuous_on: logical 
        If true, the continuous algorithm is run. Only one should be true. 

    Discrete_on: logical 
        If true, the discrete algorithm is run. Only one should be true.
   
    Optional
    ========
   
    Returns
    =======
    results: numpy array 
        Reuturns the results of the hyper-optimization
    """
    
    assert Continuous_on==False or Discrete_on==False, 'Both opimization algorithms cannot be on at the same time.'
    
    # Build list of optimization problem types 
    if Continuous_on:
        num_prob=10
        opt_funct=[of.WeldedBeam_Obj,of.PressureVessel_Obj,of.SpeedReducer_Obj,of.Spring_Obj,\
                   of.Ackley_Obj,of.DeJong_Obj,of.Easom_Obj,of.Griewank_Obj,of.Rastrigin_Obj,\
                   of.Shifted_Rastrigin_Obj,of.Shifted_Rosenbrock_Obj]
        dimension=[0,0,0,0,3,4,2,6,5,5]
        sl=[50,100,150,200,300,400,500]
        sf=[1,10,50,100,150,200]
    elif Discrete_on:
        num_prob=5
        tsp_prob=[os.path.abspath(os.getcwd())+'/TSPLIB/eil51.tsp',os.path.abspath(os.getcwd())+'/TSPLIB/st70.tsp',os.path.abspath(os.getcwd())+'/TSPLIB/pr107.tsp',os.path.abspath(os.getcwd())+'/TSPLIB/bier127.tsp',os.path.abspath(os.getcwd())+'/TSPLIB/ch150.tsp']
        dimension=[0,0,0,0,0]
        opt_fit=[426.0,675.0,44303.0,118282,6528.0]  
        opt_funct=[of.TSP_Obj,of.TSP_Obj,of.TSP_Obj,of.TSP_Obj,of.TSP_Obj]
        sl=[2,5,8,10,12,15]

    # Create list of optimization parameters
    pop=[5,15,20,25,30,40,60,80]
    fl=[0.0,0.15,0.3,0.45,0.6,0.75,0.9,1.0]
    fd=[0.0,0.05,0.125,0.2,0.3,0.5]
    fe=[0.0,0.05,0.10,0.15,0.2,0.3,0.4,0.5] 
    a=[0.5,0.9,1.2,1.5,1.98] 
    g=[0.5,1,1.5,2,3]
    

    # Initialize variables
    max_iter=25 #Number of algorithm iterations
    results=np.zeros((num_prob,len(g),3))  #!!

    for i in range(0,num_prob,1):

        for j in range(0,len(g),1):                     #!!
            history=[]   #List that contains the final timeline results from each optimization run 
            
            # Set default optimization settings
            opt_params=of.Get_Params(opt_funct[i],'Gnowee',dimension=dimension[i])
            S=util.Settings(optimal_fitness=opt_params.o)

            # Update current meta-optimization settings  
            if Continuous_on:
                #Default Settings
                S.p=25
                S.fl=0.2
                S.fd=0.2
                S.fe=0.1
                S.a=0.5
                S.sf=10
                S.s='lhc'
                S.sl=400
                # Updated settings for hyper-optimization
#                S.p=pop[j]           #!! - Ignore
                S.sl=400*25/S.p          #!! - Ignore
                S.g=g[j]          #!!
            elif Discrete_on:
                # Default Settings
                S.p=25  
                S.fl=0.2
                S.fd=0.2 
                S.fe=0.1 
                S.a=0.5
                S.gm=40
                S.sl=25
                S.em=2000000
                # Updated settings for hyper-optimization
                S.of=opt_fit[i]  
#                S.p=pop[j]           #!!    - Ignore        
                S.sl=25*40/S.p          #!! - Ignore
                S.gm=40*40/S.p          #!! - Ignore
                S.g=g[j]             #!!
            
            for k in range(0,max_iter,1):
                print "Run # %d, %d, %d" %(i,j,k)
                S.d=False

                #Run Optimization Algorithm
                if Continuous_on:
                    (timeline)=cg.Gnowee(opt_funct[i],opt_params.lb,opt_params.ub,S)
                elif Discrete_on:
                    (timeline)=dg.Gnowee(opt_funct[i],S,dg.Read_TSP(tsp_prob[i]))

                # Save final timeline data for future processing
                mingen = min(timeline,key=attrgetter('f'))
                history.append(util.Event(mingen.g,mingen.e,mingen.f,mingen.d))

            # Calculate averages and standard deviations
            tmp=[]
            averages=util.Event(sum(c.g for c in history)/float(len(history)),sum(c.e for c in history)/float(len(history)), \
                        sum(c.f for c in history)/float(len(history)),tmp)

            if max_iter >1:
                for l in range(len(history[-1].d)):
                    std_dev=util.Event(m.sqrt(sum([(c.g - averages.g)**2 for c in history])/(len(history) - 1)),\
                              m.sqrt(sum([(c.e - averages.e)**2 for c in history])/(len(history) - 1)), \
                              m.sqrt(sum([(c.f - averages.f)**2 for c in history])/(len(history) - 1)),tmp)
            else:
                tmp=np.zeros(len(history[-1].d))
                std_dev=util.Event(0,0,0.0,tmp)

            # Save Results for plotting
            results[i,j,0]=g[j]    #!!
            if S.of==0.0:
                results[i,j,1]=(averages.f*(averages.e+3*std_dev.e))
            else:
                results[i,j,1]=(abs((averages.f-S.of)/S.of)*(averages.e+3*std_dev.e))
            if S.of !=0:
                results[i,j,2]=m.sqrt((std_dev.f/averages.f)**2+(std_dev.e/averages.e)**2)/S.of*results[1,j,1]
            else:
                results[i,j,2]=m.sqrt((std_dev.f/averages.f)**2+(std_dev.e/averages.e)**2)*results[1,j,1]
 
    print repr(results)
    
    #Plot the results
    if Continuous_on:
        label=['\\textbf{Welded Beam}','\\textbf{Pressure Vessel}','\\textbf{Speed Reducer}','\\textbf{Spring}']
        title='\\textbf{Meta-Optimization of Gnowee Algorithm for Step Scaling Factor (S.sf)}'   #!! - Ignore
        op.Plot_Meta_Optimization(results[0:4],label,title)
        label=['\\textbf{Ackley}','\\textbf{De Jong}','\\textbf{Easom}','\\textbf{Griewank}','\\textbf{Rastrigin}' \
               ,'\\textbf{Rosenbrock}']
        op.Plot_Meta_Optimization(results[4:10],label,title)
    elif Discrete_on:
        label=['\\textbf{Eil51}','\\textbf{St70}','\\textbf{Pr107}','\\textbf{Bier127}','\\textbf{Ch150}']
        title='\\textbf{Meta-Optimization of Gnowee Algorithm for Step Scaling Factor (S.sf)}'  #!! - Ignore
        op.Plot_Meta_Optimization(results,label,title)
    
    return results 
    