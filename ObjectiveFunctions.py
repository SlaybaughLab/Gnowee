#######################################################################################################
#
# Module : ObjectiveFunctions.py
#
# Contains : Objective functions and supporting functions and methods that can be used to validate 
#            optimization programs.
#
# Author : James Bevins
#
# Last Modified: 13Oct15
#
#######################################################################################################

import math as m
import numpy as np
import operator 

#---------------------------------------------------------------------------------------#
def Spring_Obj(u,debug=False):
    """
    Spring objective function with constraints for penalty method
    Near Optimal Example: u=[0.05169046,0.356750, 11.287126] w/ fitness=0.0126653101469
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design    
    g : array
        The g constraint value for each design    
   
    """
    assert len(u)==3, 'Spring design needs to specify D, W, and L and only those 3 parameters.'
    assert u[0]!=0 and u[1]!=0 and u[2]!=0, "Design values cannot be zero %r" %u
    
    #Inequality constraints
    g=[]
    g.append(1.-u[1]**3*u[2]/(71785.*u[0]**4))
    g.append((4*u[1]**2-u[0]*u[1])/(12566*(u[1]*u[0]**3-u[0]**4))+1/(5108*u[0]**2)-1)
    g.append(1-140.45*u[0]/(u[1]**2*u[2]))
    g.append((u[0]+u[1])/1.5-1)
    if debug:
        print 'Inequality constraints: %r' %g
        
    #Equality constraints
    geq=[]
    if debug:
        print 'Equality constraints: %r' %geq
        
    #Evaluate fitness
    fitness=((2+u[2])*u[0]**2*u[1])+Get_Penalty(ineq=g)    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g    

#---------------------------------------------------------------------------------------# 
def WeldedBeam_Obj(u,debug=False):
    """
    Welded beam objective function with constraints for penalty method
    Taken from: "Solving Engineering Optimization Problems with the Simple Constrained Particle Swarm Optimizer"
    Near Optimal Example: u=[0.20572965, 3.47048857, 9.0366249, 0.20572965] w/ fitness=1.7248525603892848
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design
    g : array
        The g constraint value for each design  
   
    """
    assert len(u)==4, 'Beam design needs to specify 4 parameters.'
    assert u[0]!=0 and u[1]!=0 and u[2]!=0 and u[3]!=0, "Design values cannot be zero %r" %u
    
    #Inequality constraints
    em=6000.*(14+u[1]/2.)
    r=m.sqrt(u[1]**2/4.+((u[0]+u[2])/2.)**2)
    j=2.*(u[0]*u[1]*m.sqrt(2)*(u[1]**2/12.+((u[0]+u[2])/2.)**2))
    tau_p=6000./(m.sqrt(2)*u[0]*u[1])
    tau_dp=em*r/j
    tau=m.sqrt(tau_p**2+2.*tau_p*tau_dp*u[1]/(2.*r)+tau_dp**2)
    sigma=504000./(u[3]*u[2]**2)
    delta=65856000./((30*10**6)*u[3]*u[2]**2)
    pc=4.013*(30.*10**6)*m.sqrt(u[2]**2*u[3]**6/36.)/196.*(1.-u[2]*m.sqrt((30.*10**6)/(4.*(12.*10**6)))/28.)
    g=[]
    g.append(tau-13600.)
    g.append(sigma-30000.)
    g.append(u[0]-u[3])
    g.append(0.10471*u[0]**2+0.04811*u[2]*u[3]*(14.+u[1])-5.0)
    g.append(0.125-u[0])
    g.append(delta-0.25)
    g.append(6000-pc)
    if debug:
        print 'Inequality constraints: %r' %g
        
    #Equality constraints
    geq=[]
    if debug:
        print 'Equality constraints: %r' %geq
        
    #Evaluate fitness
    fitness=1.10471*u[0]**2*u[1]+0.04811*u[2]*u[3]*(14.0+u[1])+Get_Penalty(ineq=g)     
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g 

#---------------------------------------------------------------------------------------#           
def PressureVessel_Obj(u,debug=False):
    """
    Pressure vessel objective function with constraints for penalty method
    Taken from: "Solving Engineering Optimization Problems with the Simple Constrained Particle Swarm Optimizer"
    Near Optimal Example: u=[0.81250000001, 0.4375,42.098445595854923, 176.6365958424394] w/ fitness=6059.714335 
    Optimal obtained using CS: u=[0.7781686880924992, 0.3846491857203429, 40.319621144688995, 199.99996630362293]
       w/ fitness=5885.33285347
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design
    g : array
        The g constraint value for each design  
   
    """
    assert len(u)==4, 'Pressure vessel design needs to specify 4 parameters.'
    assert u[0]!=0 and u[1]!=0 and u[2]!=0 and u[3]!=0, "Design values cannot be zero %r" %u
    
    #Inequality constraints
    g=[]
    g.append(-u[0]+0.0193*u[2])
    g.append(-u[1]+0.00954*u[2])
    g.append(-np.pi*u[2]**2*u[3]-4./3.*np.pi*u[2]**3+1296000)
    g.append(u[3]-240)
    if debug:
        print 'Inequality constraints: %r' %g
        
    #Equality constraints
    geq=[]
    if debug:
        print 'Equality constraints: %r' %geq
        
    fitness=0.6224*u[0]*u[2]*u[3]+1.7781*u[1]*u[2]**2+3.1661*u[0]**2*u[3]+19.84*u[0]**2*u[2]+Get_Penalty(ineq=g)     
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g  

#---------------------------------------------------------------------------------------# 
def SpeedReducer_Obj(u,debug=False):
    """
    Speed Reducer objective function with constraints for penalty method
    Taken from: "Solving Engineering Optimization Problems with the Simple Constrained Particle Swarm Optimizer"
    Near Optimal Example: u=[3.500000, 0.7, 17, 7.300000, 7.800000, 3.350214, 5.286683] w/ fitness=2996.34784914
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design
    g : array
        The g constraint value for each design  
   
    """
    assert len(u)==7, 'Spead reducer design needs to specify 7 parameters.'
    assert u[0]!=0 and u[1]!=0 and u[2]!=0 and u[3]!=0 and u[4]!=0 and u[5]!=0 and u[6]!=0, \
         "Design values cannot be zero %r" %u
    
    #Inequality constraints
    g=[]
    g.append(27./(u[0]*u[1]**2*u[2])-1.)
    g.append(397.5/(u[0]*u[1]**2*u[2]**2)-1.)
    g.append(1.93*u[3]**3/(u[1]*u[2]*u[5]**4)-1.)
    g.append(1.93*u[4]**3/(u[1]*u[2]*u[6]**4)-1.)
    g.append(1.0/(110.*u[5]**3)*m.sqrt((745.0*u[3]/(u[1]*u[2]))**2+16.9*10**6)-1)
    g.append(1.0/(85.*u[6]**3)*m.sqrt((745.0*u[4]/(u[1]*u[2]))**2+157.5*10**6)-1)
    g.append(u[1]*u[2]/40.-1)
    g.append(5.*u[1]/u[0]-1)
    g.append(u[0]/(12.*u[1])-1)
    g.append((1.5*u[5]+1.9)/u[3]-1)
    g.append((1.1*u[6]+1.9)/u[4]-1)
    
    if debug:
        print 'Inequality constraints: %r' %g
        
    #Equality constraints
    geq=[]
    if debug:
        print 'Equality constraints: %r' %geq
        
    #Evaluate fitness
    fitness=0.7854*u[0]*u[1]**2*(3.3333*u[2]**2+14.9334*u[2]-43.0934) \
            - 1.508*u[0]*(u[5]**2+u[6]**2) + 7.4777*(u[5]**3+u[6]**3) \
            + 0.7854*(u[3]*u[5]**2+u[4]*u[6]**2)+Get_Penalty(ineq=g)     
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g 

#---------------------------------------------------------------------------------------# 
def Ackley_Obj(u,debug=False):
    """
    Ackley Function: Mulitmodal, n dimensional
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[0, 0, 0, 0, ... n] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1, 'The De Jong Function must be at least of dimension 1.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=-20*m.exp(-1./5.*m.sqrt(1./len(u)*sum(u[i]**2 for i in range(len(u))))) \
            - m.exp(1./len(u)*sum(m.cos(2*m.pi*u[i]) for i in range(len(u)))) + 20 + m.exp(1)
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Shifted_Ackley_Obj(u,debug=False):
    """
    Ackley Function: Mulitmodal, n dimensional
    Ackley Function that is shifted from the symmetric 0,0,..n optimimum to where 
    the optimum for each dimension occurs at that dimension number.
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[0, 1, 2, 3, ... n] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1 and len(u)<=25, 'The Shifted Ackley Function must have dimensions between 1 and 25.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=-20*m.exp(-1./5.*m.sqrt(1./len(u)*sum((u[i]-i)**2 for i in range(len(u))))) \
            - m.exp(1./len(u)*sum(m.cos(2*m.pi*(u[i]-i)) for i in range(len(u)))) + 20 + m.exp(1)
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def DeJong_Obj(u,debug=False):
    """
    De Jong Function: Unimodal, n dimensional
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[0, 0, 0, 0, ... n] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1, 'The De Jong Function must be at least of dimension 1.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=sum(i**2 for i in u) 
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Shifted_DeJong_Obj(u,debug=False):
    """
    De Jong Function: Unimodal, n dimensional
    A De Jong Function that is shifted from the symmetric 0,0,..n optimimum to where 
    the optimum for each dimension occurs at that dimension number.
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[0, 1, 2, 3, ... n-1] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1 and len(u)<=5, 'The Shifted De Jong Function must have dimensions between 1 and 5.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=sum((u[i]-i)**2 for i in range(len(u))) 
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Easom_Obj(u,debug=False):
    """
    Easom Function: Mulitmodal, 2-dimensional
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[pi,pi] w/ fitness=-1.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)==2, 'The Easom Function must have a dimension of 2.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=-m.cos(u[0])*m.cos(u[1])*m.exp(-(u[0]-m.pi)**2-(u[1]-m.pi)**2) 
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Shifted_Easom_Obj(u,debug=False):
    """
    Easom Function: Mulitmodal, 2-dimensional
    Easom Function that is shifted from the symmetric pi, pi optimimum 
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[pi,pi+1] w/ fitness=-1.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)==2, 'The Easom Function must have a dimension of 2.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=-m.cos(u[0])*m.cos(u[1]-1)*m.exp(-(u[0]-m.pi)**2-(u[1]-1-m.pi)**2)
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Griewank_Obj(u,debug=False):
    """
    Griewank Function: Mulitmodal, n-dimensional
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[0,0,0....0] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1 and len(u)<=600, 'The Griewank Function must have a dimension between 1 and 600.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=1./4000.*sum((u[i])**2 for i in range(len(u))) - prod(m.cos(u[i]/m.sqrt(i+1)) for i in range(len(u))) +1.
   
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Shifted_Griewank_Obj(u,debug=False):
    """
    Griewank Function: Mulitmodal, n-dimensional
    Griewank Function that is shifted from the symmetric 0,0,0...0 optimimum 
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[0,1,2,...n-1] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1 and len(u)<=600, 'The Griewank Function must have a dimension between 1 and 600.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=1./4000.*sum((u[i]-i)**2 for i in range(len(u))) - prod(m.cos((u[i]-i)/m.sqrt(i+1)) for i in range(len(u))) +1.
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Rastrigin_Obj(u,debug=False):
    """
    Rastrigin Function: Mulitmodal, n-dimensional
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[0,0,0....0] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1 and len(u)<=5, 'The Rastrigin Function must have a dimension between 1 and 5.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=10.*len(u)+sum((u[i])**2 -10.*np.cos(2.*np.pi*u[i]) for i in range(len(u))) 
   
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Shifted_Rastrigin_Obj(u,debug=False):
    """
    Rastrigin Function: Mulitmodal, n-dimensional
    Rastrigin Function that is shifted from the symmetric 0,0,0...0 optimimum 
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[0,1,2,...n-1] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1 and len(u)<=5, 'The Rastrigin Function must have a dimension between 1 and 5.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=10.*len(u)+sum((u[i]-i)**2 -10.*np.cos(2.*np.pi*(u[i]-i)) for i in range(len(u)))
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Rosenbrock_Obj(u,debug=False):
    """
    Rosenbrock Function: uni-modal, n-dimensional
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[1,1,1....1] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1 and len(u)<=5, 'The Rosenbrock Function must have a dimension between 1 and 5.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=sum((u[i]-1)**2 +100.*(u[i+1]-u[i]**2)**2 for i in range(len(u)-1))
   
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def Shifted_Rosenbrock_Obj(u,debug=False):
    """
    Rosenbrock Function: uni-modal, n-dimensional
    Rosenbrock Function that is shifted from the symmetric 0,0,0...0 optimimum 
    Taken from: "Nature-Inspired Optimization Algorithms"
    Optimal Example: u=[1,2,3,...n] w/ fitness=0.0
   
    Parameters
    ==========
    u : array
        The design parameters to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design   
    """
    assert len(u)>=1 and len(u)<=5, 'The Rosenbrock Function must have a dimension between 1 and 5.'
    
    #Inequality constraints
    g=[]
    
    #Evaluate fitness
    fitness=sum((u[i]-1-i)**2 +100.*((u[i+1]-(i+1))-(u[i]-i)**2)**2 for i in range(len(u)-1))
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,g

#---------------------------------------------------------------------------------------# 
def TSP_Obj(u,debug=False):
    """
    Generic objective funtion to evaluate the TSP optimization by calculating total distance traveled
   
    Parameters
    ==========
    u : array
        The city pairs to be evaluated
   
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    fitness : array
        The fitness of each design 
   
    """
        
    #Evaluate fitness
    fitness=0
    for i in range(1,len(u),1):
        fitness=fitness+round(m.sqrt((u[i][0]-u[i-1][0])**2 + (u[i][1]-u[i-1][1])**2))
    
    #Complete the tour
    fitness=fitness+round(m.sqrt((u[0][0]-u[-1][0])**2 + (u[0][1]-u[-1][1])**2))
    
    if debug:
        print 'Evaluated fitness: %f' %fitness    
    
    return fitness,0.0

#---------------------------------------------------------------------------------------# 
def Get_Penalty(ineq=[],eq=[],debug=False):
    """
    Calculate the constraint violation penalty, if any
   
    Parameters
    ==========
   
    Optional
    ========   
    ineq : array
        Evaluated inequality constraints
        (Default: empty array)
    eq : array
        Evaluated equality constraints
        (Default: empty array)     
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    Returns
    =======
    penalty: array
        The fitness of each design
   
    """
    penalty=0.    #Total constraint violation penalty
    pen=1E20   #Per constraint violation penalty
    
    #Apply inequality constraints
    for i in ineq:
        penalty=penalty+pen*i**2*Get_H(i)
    if debug:
        print 'Inequality constraints violation penalty: %f' %penalty
        
    #Apply equality constraints
    for i in eq:
        penalty=penalty+pen*i**2*Get_Heq(i)
    if debug:
        print 'Equality constraints violation penalty: %f' %penalty
    
    return penalty

#---------------------------------------------------------------------------------------#           

def Get_H(g,debug=False):
    """
    Tests inequality constraints
   
    Parameters
    ==========
    g : array
        Evaluated inequality constraints
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    Returns
    =======
    H: integer
        Value representing if constraints were violated or not
    """
    
    if g < .0:
        H=0.
    else:
        H=1.
    if debug:
        print 'H= %d' %H
    return H
            
#---------------------------------------------------------------------------------------#           
def Get_Heq(geq,debug=False):
    """
    Tests inequality constraints
   
    Parameters
    ==========
    geq : array
        Evaluated equality constraints
    Optional
    ========   
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    Returns
    =======
    H: integer
        Value representing if constraints were violated or not
    """
    if geq == 0.:
        H=0.
    else:
        H=1.
    if debug:
        print 'H= %d' %H
    return H   

#---------------------------------------------------------------------------------------#           
class switch(object):
    """
    Creates a switch class object to switch between cases
   
    Parameters
    ==========
    value : string
        case selector value
    Returns
    =======
    True or False based on match
    """
    
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: 
            self.fall = True
            return True
        else:
            return False
                 
#---------------------------------------------------------------------------------------#  
class Parameters:
    """
    Creates an parameter object containing key features of the chosen optimization 
    problem type
   
    Parameters
    ==========
    lower_bounds : array
        The lower bounds of the design variable(s)
    upper_bounds : array
        The upper bounds of the design variable(s)
    optimum : array
        The global optimal solution obtained from "Solving Engineering Optimization Problems with the Simple
        Constrained Particle Swarm Optimizer"
    label : string array
        The y axis labels 
    plt_title : string
        The plot title 
    hist_title : string
        The plot title for the histogram
    Returns
    =======
    None
    """
        
    def __init__(self,lower_bounds,upper_bounds,optimum,label,plt_title,hist_title):
        self.lb=lower_bounds
        self.ub=upper_bounds
        self.o=optimum
        self.l=label
        self.pt=plt_title
        self.ht=hist_title
        
#---------------------------------------------------------------------------------------#  
def Get_Params(funct,opt,dimension=0,debug=False):
    """
    Gets the parameters associated with running and outputting an optimization run given 
    a type of objective function.
   
    Parameters
    ==========
    funct : Function
        Name of function being optimized 
    opt : string 
        Name of optimization program used
    Optional
    ========   
    dimension : integer 
        Used to set the dimension for scalable problems
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    Returns
    =======
    params : object
        Contains key features of the chosen optimization problem type
    """       
        
    for case in switch(funct):
        if case(Spring_Obj): 
            params=Parameters(np.array([0.05, 0.25, 2.0]),np.array([2.0, 1.3, 15.0]),0.012665, \
                              ['\\textbf{Fitness}','\\textbf{Width}','\\textbf{Diameter}','\\textbf{Length}'], \
                              '\\textbf{Spring Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Spring Optimization using %s}' %opt)
            break
        if case(PressureVessel_Obj):
            params=Parameters(np.array([0.0625, 0.0625, 10.0, 1E-8]),np.array([1.25, 99*0.0625, 50.0, 200.0]), \
                              6059.714335, ['\\textbf{Fitness}','\\textbf{Thickness}','\\textbf{Head Thickness}', \
                                                  '\\textbf{Inner Radius}','\\textbf{Cylinder Length}'], \
                              '\\textbf{Pressure Vessel Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Pressure Vessel Optimization using %s}' %opt)
            break
        if case(WeldedBeam_Obj):
            params=Parameters(np.array([0.1, 0.1, 1E-8, 1E-8]),np.array([10.0, 10.0, 10.0, 2.0]),1.724852, \
                              ['\\textbf{Fitness}','\\textbf{Weld H}','\\textbf{Weld L}','\\textbf{Beam H}',\
                               '\\textbf{Beam W}'],\
                              '\\textbf{Welded Beam Optimization using %s}' %opt,\
                              '\\textbf{Function Evaluations for Welded Beam Optimization using %s}' %opt)
            break
        if case(SpeedReducer_Obj):
            params=Parameters(np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0]), \
                              np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]), 2996.348165, \
                              ['\\textbf{Fitness}','\\textbf{Face Width}','\\textbf{Module}','\\textbf{Pinion Teeth}', \
                                     '\\textbf{1st Shaft L}','\\textbf{2nd Shaft L}','\\textbf{1st Shaft D}', \
                                     '\\textbf{2nd Shaft D}'], \
                              '\\textbf{Speed Reducer Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Speed Reducer Optimization using %s}' %opt)
            break
        if case(DeJong_Obj):
            params=Parameters(np.ones(dimension)*-5.12,np.ones(dimension)*5.12, 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{De Jong Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for De Jong Function Optimization using %s}' %opt)
            break
        if case(Shifted_DeJong_Obj):
            params=Parameters(np.ones(dimension)*-5.12,np.ones(dimension)*-5.12, 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Shifted De Jong Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Shifted De Jong Function Optimization using %s}' %opt)
            break
        if case(Ackley_Obj):
            params=Parameters(np.ones(dimension)*-25.,np.ones(dimension)*25., 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Ackley Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Ackley Function Optimization using %s}' %opt)
            break
        if case(Shifted_Ackley_Obj):
            params=Parameters(np.ones(dimension)*-25.,np.ones(dimension)*25., 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Shifted Ackley Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Shifted Ackley Function Optimization using %s}' %opt)
            break
        if case(Easom_Obj):
            params=Parameters(np.array([-100.,-100.]),np.array([100.,100.]), -1.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{x}','\\textbf{y}'],\
                              '\\textbf{Easom Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Easom Function Optimization using %s}' %opt)
            break
        if case(Shifted_Easom_Obj):
            params=Parameters(np.array([-100.,-100.]),np.array([100.,100.]), -1.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{x}','\\textbf{y}'],\
                              '\\textbf{Shifted Easom Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Shifted Easom Function Optimization using %s}' %opt)
            break
        if case(Griewank_Obj):
            params=Parameters(np.ones(dimension)*-600.,np.ones(dimension)*600., 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Griewank Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Griewank Function Optimization using %s}' %opt)
            break
        if case(Shifted_Griewank_Obj):
            params=Parameters(np.ones(dimension)*-600.,np.ones(dimension)*600., 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Shifted Easom Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Shifted Easom Function Optimization using %s}' %opt)
            break
        if case(Rastrigin_Obj):
            params=Parameters(np.ones(dimension)*-5.12,np.ones(dimension)*5.12, 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Rastrigin Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Rastrigin Function Optimization using %s}' %opt)
            break
        if case(Shifted_Rastrigin_Obj):
            params=Parameters(np.ones(dimension)*-5.12,np.ones(dimension)*5.12, 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Shifted Rastrigin Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Shifted Rastrigin Function Optimization using %s}' %opt)
            break
        if case(Rosenbrock_Obj):
            params=Parameters(np.ones(dimension)*-5.,np.ones(dimension)*5., 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Rosenbrock Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Rosenbrock Function Optimization using %s}' %opt)
            break
        if case(Shifted_Rosenbrock_Obj):
            params=Parameters(np.ones(dimension)*-5.,np.ones(dimension)*5., 0.0000, \
                              ['\\textbf{Fitness}'] + ['\\textbf{Dim \#%s}' %i for i in range(dimension)],\
                              '\\textbf{Shifted Rosenbrock Function Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for Shifted Rosenbrock Function Optimization using %s}' %opt)
            break
        if case(TSP_Obj):
            params=Parameters([], [], 0.0,['\\textbf{Fitness}'],'\\textbf{TSP Optimization using %s}' %opt, \
                              '\\textbf{Function Evaluations for TSP using %s}' %opt)
            break
        if case(): # default, could also just omit condition or 'if True'
            print "something else!"
            # No need to break here, it'll stop anyway
            
    return params

#---------------------------------------------------------------------------------------#  
def prod(iterable):
    """
    Computes the product of a set of numbers (ie big PI, mulitplicative equivalent to sum)
   
    Parameters
    ==========
    iterable : list or array or generator
        Iterable set to multiply 
    Optional
    ========   
    
    Returns
    =======
     : scalar
        The product of all of the items in iterable
    """    
    return reduce(operator.mul, iterable, 1)         
