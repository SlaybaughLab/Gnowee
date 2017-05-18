function Results_multistart=ssm_multistart(problem,opts,varargin)

% Function   : ssm_multistart
% Written by : Process Engineering Group IIM-CSIC (jegea@iim.csic.es)
% Created on : 01/07/2005
% Last Update: 20/04/2007
%
% Script to do multistart optimization (e.g. Multiple local optimization
% starting from different initial points uniformly distributed withing the
% bounds)
%
%
%          Results_multistart=ssm_multistart(problem,opts,p1,p2,...,pn)
%
% Input Parameters:
%         problem   - Structure containing problem
%         
%               problem.f               = Name of the file containing the objective function
%               problem.x_L             = Lower bounds of decision variables
%               problem.x_U             = Upper bounds of decision variables
%               problem.x_0             = Initial point(s) (optional)      
%   
%               Additionally, fill the following fields if your problem has
%               non-linear constraints
%
%               problem.neq             = Number of equality constraints (do not define it if there are no equality constraints)
%               problem.c_L             = Lower bounds of nonlinear constraints
%               problem.c_U             = Upper bounds of nonlinear constraints
%               problem.int_var         = Number of integer variables
%               problem.bin_var         = Number of binary variables
%               NOTE: The order of decision variables is x=[cont int bin]
%
%         opts      - Structure containing options
%
%           opts.ndiverse: Number of uniformly distributed points inside the bounds to do the multistart optimization (default 100)
%
%           Local options
%               opts.local.solver             = Choose local solver ( default 'fmincon')
%                                                     0: Local search deactivated
%                                                     'fmincon' (Mathworks)
%                                                     'clssolve' (Tomlab)
%                                                     'snopt' (Tomlab)
%                                                     'nomad' (Abramson)
%                                                     'npsol' (Tomlab)
%                                                     'solnp' (Ye)
%                                                     'n2fb' (For parameter stimation problems. Single precision)
%                                                     'dn2fb' (For parameter stimation problems. Double precision)
%                                                     'dhc' (Yuret) Direct search
%                                                     'fsqp'
%                                                     'ipopt'
%                                                     'misqp' (Exler) For mixed integer problems
%               opts.local.tol                = Level of tolerance in local search
%               opts.local.iterprint          = Print each iteration of local solver on screen
%
%   p1,p2... :  optional input parameters to be passed to the objective
%   function
%
%
% Output Parameters:
%         Results_multistart      -Structure containing results
%
%               Results_multistart.fbest       :Best objective function value found after the multistart optimization
%               Results_multistart.xbest       :Vector providing the best function value
%               Results_multistart.x0          :Array containing the vectors used for the multistart optimization (in rows)
%               Results_multistart.f0          :Vector containing the objective function values of the initial solutions used in the multistart optimization
%               Results_multistart.func        :Vector containing the objective function values obtained after every local search
%               Results_multistart.xxx         :Array containing the vectors provided by the local optimization (in rows)
%               Results_multistart.no_conv     :Array containing the initial points that did not provide any solution when the local search was applied (in rows)
%               Results_multistart.nfuneval    :Vector containing the number of function evaluations in every optimisation
%               Results_multistart.time        :Total CPU time to carry out the multistart optimization           

%               NOTE: To plot an histogram of the results: hist(Results_multistart.func)

global input_par
rand('state',sum(100*clock))


cpu_time=cputime;

%Extra input parameters for fsqp and n2fb
if nargin>2
    if  strcmp(opts.local.solver,'dn2fb') | strcmp(opts.local.solver,'n2fb') |...
            strcmp(opts.local.solver,'fsqp') | strcmp(opts.local.solver,'nomad') 
        input_par=varargin;
    end
end


x_U=problem.x_U;
x_L=problem.x_L;

if not(isfield(problem,'neq')) | isempty(problem.neq)
    neq=0;
else
    neq=problem.neq;
end

if not(isfield(problem,'x_0'))
    x_0=[];
else
    x_0=problem.x_0;
end


if not(isfield(problem,'c_U'))
    c_U=[];
    c_L=[];
else
    c_U=problem.c_U;
    c_L=problem.c_L;
end

if not(isfield(problem,'int_var')) | isempty(problem.int_var)
    int_var=0;
else
    int_var=problem.int_var;
end

if not(isfield(problem,'bin_var')) | isempty(problem.bin_var)
    bin_var=0;
else
    bin_var=problem.bin_var;
end

if not(isfield(problem,'ndata'))
    ndata=[];
else
    ndata=problem.ndata;
end


%Load default values
default=ssm_defaults;

%Set all options
opts=ssm_optset(default,opts);

weight=opts.weight;
tolc=opts.tolc;
local_solver=opts.local.solver;


%Check if bounds have the same dimension
if length(x_U)~=length(x_L)
    disp('Upper and lower bounds have different dimension!!!!')
    disp('EXITING')
    Results_multistart=[];
    return
else
    %Number of decision variables
    nvar=length(x_L);                                    
end

%Transformacion de variables de cadena en funciones para que la llamada con
%feval sea mas rapida
fobj=str2func(problem.f);

if (int_var+bin_var) & local_solver & not(strcmp(local_solver,'misqp'))
     fprintf('For problems with integer and/or binary variables you must use MISQP as a local solver \n');
        fprintf('EXITING \n');
        Results_multistart=[];
        return
end


if local_solver & strcmp(local_solver,'n2fb') | strcmp(local_solver,'dn2fb')
    n_out_f=nargout(problem.f);
    if n_out_f<3
        fprintf('%s requires 3 output arguments \n',local_solver);
        fprintf('EXITING \n');
        Results_multistart=[];
        return
    else
        %Generate a random point within the bounds
        randx=rand(1,nvar).*(x_U-x_L)+x_L;
        [f g R]=feval(fobj,randx,varargin{:});
        ndata=length(R);
    end
else 
    ndata=[];
end


%If there are equality constraints
if neq
    %Set the new bounds for all the constraints
    c_L=[-tolc*ones(1,neq) c_L];
    c_U=[tolc*ones(1,neq) c_U];
end

nconst=length(c_U);
if nconst
    n_out_f=nargout(problem.f);
    if n_out_f<2
        fprintf('For constrained problems the objective function must have at least 2 output arguments \n');
        fprintf('EXITING \n');
        Results_multistart=[];
        return
    end
end

%Cargamos los archivos auxiliares
ssm_aux_local(problem.f,x_L,x_U,c_L,c_U,neq,local_solver,nvar,varargin{:});


func=[];
xxx=[];
no_conv=[];
x0=[];
nfuneval=[];



if opts.ndiverse
    multix=rand(opts.ndiverse,nvar);

    %Put the variables inside the bounds
    a=repmat(x_U-x_L,[opts.ndiverse,1]);
    b=repmat(x_L,[opts.ndiverse,1]);
    multix=multix.*a+b;
else
    multix=[];
end

multix=[x_0; multix];

if (int_var) | (bin_var)
    multix=ssm_round_int(multix,int_var+bin_var,x_L,x_U);
end


for i=1:size(multix,1)
    f0(i)=feval(problem.f,multix(i,:),varargin{:});
end

for i=1:size(multix,1)
    x0=multix(i,:);

    if opts.local.iterprint
        fprintf('\n');
        fprintf('Local search number: %i \n',i);
        fprintf('Call local solver: %s \n', upper(local_solver))
        fprintf('Initial point function value: %f \n',f0(i));

        tic
    end
    [x,fval,exitflag,numeval]=ssm_localsolver(x0,x_L,x_U,c_L,c_U,neq,ndata,int_var,bin_var,fobj,...
        local_solver,opts.local.iterprint,opts.local.tol,weight,nconst,tolc,varargin{:});

    if opts.local.iterprint
        fprintf('Local solution function value: %f \n',fval);

        if exitflag<0
            fprintf('Warning!!! This could be an infeasible solution: \n');
        end

        fprintf('Number of function evaluations in the local search: %i \n',numeval);
        fprintf('CPU Time of the local search: %f  seconds \n\n',toc);
    end

    if exitflag<0 |isnan(fval) | isinf(fval)
        no_conv=[no_conv;x0];
    else
        func=[func; fval];
        xxx=[xxx;x];
    end
    nfuneval=[nfuneval;numeval];
end
hist(func,100);
xlabel('Objective Function Value');
ylabel('Frequency');
title('Histogram')
[aaa iii]=min(func);
Results_multistart.x0=multix;
Results_multistart.f0=f0;
Results_multistart.func=func;
Results_multistart.xxx=xxx;
Results_multistart.fbest=aaa;
Results_multistart.xbest=xxx(iii,:);
Results_multistart.no_conv=no_conv;
Results_multistart.nfuneval=nfuneval;
Results_multistart.time=cputime-cpu_time;

save ssm_multistart_report problem opts Results_multistart

ssm_delete_files(local_solver,c_U);

return
