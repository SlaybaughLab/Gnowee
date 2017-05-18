clear all

%========================= PROBLEM SPECIFICATIONS ===========================
nvar=5;
problem.x_L=-5*ones(1,nvar);
problem.x_U=5*ones(1,nvar);
problem.f='rosenbrock';

opts.maxeval=10000;
algorithm='ESS';

[Results]=MEIGO(problem,opts,algorithm);
