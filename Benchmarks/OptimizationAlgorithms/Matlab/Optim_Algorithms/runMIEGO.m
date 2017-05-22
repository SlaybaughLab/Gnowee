clear all
clc

%========================= WELDED BEAM PROBLEM SPECIFICATIONS ===========================
% problem.f='weldedbeam_fun';
% nvar=4;
% problem.x_L=([0.1 0.1 1E-8 1E-8]);
% problem.x_U=([10.0 10.0 10.0 2.0]);
% problem.int_var=0;
% optim=1.724852;

%========================= PRESSURE VESSEL SPECIFICATIONS ===========================
% problem.f='pressurevessel_fun';
% nvar=4;
% problem.x_L=([0.0625 0.0625 10.0 1E-8]);
% problem.x_U=([99*0.0625 99*0.0625 50.0 200.0]);
% problem.int_var=0;
% optim=5885.332800;

%========================= SPEED REDUCER SPECIFICATIONS ===========================
% problem.f='speedreducer_fun';
% nvar=7;
% problem.x_L=([2.6 0.7 17.0 7.3 7.8 2.9 5.0]);
% problem.x_U=([3.6 0.8 28.0 8.3 8.3 3.9 5.5]);
% problem.int_var=0;
% optim=2996.348165;

%========================= SPRING PROBLEM SPECIFICATIONS ===========================
% problem.f='spring_fun';
% nvar=3;
% problem.x_L=([0.05 0.25 2.0]);
% problem.x_U=([2.0 1.3 15.0]);
% problem.int_var=0;
% optim=0.012665;

%========================= MI SPRING PROBLEM SPECIFICATIONS ===========================
% problem.f='mi_spring';
% nvar=3;
% problem.x_L=([0.01 1 1]);
% problem.x_U=([3.0 10 42]);
% problem.int_var=2;
% optim=2.65872;

%========================= MI PRESSURE VESSEL PROBLEM SPECIFICATIONS ===========================
% problem.f='mi_pressure_vessel';
% nvar=4;
% % [R,L,t_s,t_h]
% problem.x_L=([10.0 1E-8 1 1]);
% problem.x_U=([50.0 200.0 99 99 ]);
% problem.int_var=2;
% optim=6059.714335;

%========================= CHEMICAL PROCESS PROBLEM SPECIFICATIONS ===========================
% problem.f='mi_chemical_process';
% nvar=7;
% % [x1,x2,x3,y1,y2,y3,y4]
% problem.x_L=([0,0,0,0,0,0,0]);
% problem.x_U=([10.0,10.0,10.0,1,1,1,1]);
% problem.int_var=4;
% optim=4.579582;

%========================= ACKLEY PROBLEM SPECIFICATIONS ===========================
% problem.f='ackley_fun';
% nvar=3;
% problem.x_L=([-25.0 -25.0 -25.0]);
% problem.x_U=([25.0 25.0 25.0]);
% problem.int_var=0;
% optim=0.0;

%========================= DE JONG PROBLEM SPECIFICATIONS ===========================
% problem.f='dejong_fun';
% nvar=4;
% problem.x_L=([-5.12 -5.12 -5.12 -5.12]);
% problem.x_U=([5.12 5.12 5.12 5.12]);
% problem.int_var=0;
% optim=0.0;

% %========================= EASOM PROBLEM SPECIFICATIONS ===========================
problem.f='easom_fun';
nvar=2;
problem.x_L=([-100.0 -100.0]);
problem.x_U=([100.0 100.0]);
problem.int_var=0;
optim=-1.0;

%========================= GREIWANK PROBLEM SPECIFICATIONS ===========================
% problem.f='griewank_fun';
% nvar=6;
% problem.x_L=([-600.0 -600.0 -600.0 -600.0 -600.0 -600.0]);
% problem.x_U=([600.0 600.0 600.0 600.0 600.0 600.0]);
% problem.int_var=0;
% optim=0.0;

%========================= RASTRIGIN PROBLEM SPECIFICATIONS ===========================
% problem.f='rastrigin_fun';
% nvar=5;
% problem.x_L=([-5.12 -5.12 -5.12 -5.12 -5.12]);
% problem.x_U=([5.12 5.12 5.12 5.12 5.12]);
% problem.int_var=0;
% optim=0.0;

%========================= ROSENBROCK PROBLEM SPECIFICATIONS ===========================
% problem.f='rosenbrock_fun';
% nvar=5;
% problem.x_L=([-5.0 -5.0 -5.0 -5.0 -5.0]);
% problem.x_U=([5.0 5.0 5.0 5.0 5.0]);
% problem.int_var=0;
% optim=0.0;

%========================= BENCHMARK SPECIFICATIONS ===========================
iter=100;
opts.maxeval=200000;
opts.ndiverse=50; 
opts.n_stuck=250;
conTol=1E-2;

%========================= EVALLIM CONVERGENCE SPECIFICATIONS ===========================
opts.maxeval=10000;%200000;
opts.n_stuck=10000;
conTol=1E-16;

%========================= SOLVER SPECIFICATIONS ===========================
opts.local.solver=0;
algorithm='eSS';
opts.inter_save=1;

% Initialize variables
fitness=[];
feval=[];
design=[];
bestf=1E15;
feval_history = zeros(2,4);

% Run Optimization
for i=1:iter
    [Results]=MEIGO(problem,opts,algorithm);
    
    for j=1:length(Results.f) 
        if optim ~= 0
           if abs(optim-Results.f(j)) <= abs(conTol*Results.f(j))
               break
           end
        else
            if abs(optim-Results.f(j)) <= conTol
               break
            end
        end
    end
    
    % Save the run results
    fitness(i)=Results.f(j);
    feval(i)=Results.neval(j);    
    design(i,:)=Results.x(end,:);
    
    % Trim arrays to only include pertinent data
    Results.neval=Results.neval(1:j);
    Results.f=Results.f(1:j);
    
    % Save best design information
    if i==1 | fitness(i)<bestf
        bestdesign=design(i,:);
        bestf=fitness(i);
        beste=feval(i);
    end
    
    % Save function eval history results to a predefined binned
    % structure; Compute the average and standard deviation using a recurrence relation
    k=2;
    for j=1:length(Results.neval)
        if j > length(feval_history)
            feval_history = [feval_history; [0,0.0,0.0,0]];
        end
        feval_history(j,1)=500*(j-1);
        while Results.neval(k) < feval_history(j,1) && k+1<=length(Results.neval) 
             k=k+1;
        end
        if k+1 > length(Results.neval) && Results.neval(k) < feval_history(j,1)
             % Initialize the array on the first run
             if feval_history(j,4)==0
                 feval_history(j,2)=Results.f(k);
                 feval_history(j,3)=0.0;
                 feval_history(j,4)=feval_history(j,4)+1;
             else
                 old_mean=feval_history(j,2);
                 feval_history(j,4)=feval_history(j,4)+1;
                 feval_history(j,2)=feval_history(j,2)+(Results.f(k)-feval_history(j,2))/feval_history(j,4);
                 feval_history(j,3)=feval_history(j,3)+(Results.f(k)-old_mean)*(Results.f(k)-feval_history(j,2));
             end
             break 
        else
             % Initialize the array on the first run
             if feval_history(j,4)==0
                 feval_history(j,2)=Results.f(k);
                 feval_history(j,3)=0.0;
                 feval_history(j,4)=feval_history(j,4)+1;
             else
                 old_mean=feval_history(j,2);
                 feval_history(j,4)=feval_history(j,4)+1;
                 feval_history(j,2)=feval_history(j,2)+(Results.f(k-1)-feval_history(j,2))/feval_history(j,4);
                 feval_history(j,3)=feval_history(j,3)+(Results.f(k-1)-old_mean)*(Results.f(k-1)-feval_history(j,2));
             end
        end
    end
end

% Compile results for output
% Output Average Solutions
fprintf('\nThe Average Optimized Solution:\n')
fprintf('------------------------------------\n')
fprintf('Design:\n')
for i=1:length(problem.x_L)
   fprintf('    var%d: %4.6f $\\mypm$ %4.5f \n', i, mean(design(:,i)), std(design(:,i)))
end
fprintf('Fitness:  %4.6f $\\mypm$ %4.5f \n',mean(fitness),std(fitness)) 
fprintf('Funct Evals: %d $\\mypm$ %d \n',mean(feval),std(feval))
if optim==0.0
    fprintf('The performance metric is %4.1f\n', mean(fitness)*(mean(feval)+3*std(feval)))
else
    fprintf('The performance metric is %4.1f\n', abs((mean(fitness)-optim)/optim)*(mean(feval)+3*std(feval)))
end
   
% Output Best solutions
fprintf('\nThe Best Optimized Solution:\n')
fprintf('------------------------------------\n')
fprintf('Design:\n')
for i=1:length(problem.x_L)
   fprintf('    var%d: %4.6f \n', i, bestdesign(i))
end
fprintf('Fitness:  %4.6f \n',bestf) 
fprintf('Funct Evals: %d  \n',beste)
if optim==0.0
    fprintf('The performance metric is %4.1f\n', bestf*beste)
else
    fprintf('The performance metric is %4.1f\n', abs((bestf-optim)/optim)*beste)
end

% Plot Histogram
figure(2);
h=histogram(feval,100,'Normalization','probability');
set(h,'FaceColor','k','EdgeColor','k');
xlabel('# Function Evals')
ylabel('Probability')
title('Function Evaluations for MI Chemical Process Optimization using MEIGO')
      
% Calculate % Diff, standard deviation, and output feval history for use in plotting in python
fprintf('np.array([');
for i=1:length(feval_history)
    % Calculate standard deviation
    if feval_history(i,3)~=0
        feval_history(i,3)=sqrt(feval_history(i,3)./(feval_history(i,4)-1))*100;
    end
    % Calculate % diff
    if optim==0.0
        feval_history(i,2)=abs(feval_history(i,2))*100;
    else
        feval_history(i,2)=abs((feval_history(i,2)-optim))/optim*100;
    end
    % Output data 
    if i==length(feval_history)
         fprintf('[%d,%f,%e, %d]', feval_history(i,1),feval_history(i,2),feval_history(i,3),feval_history(i,4));
    else
         fprintf('[%d,%f,%e, %d],', feval_history(i,1),feval_history(i,2), feval_history(i,3), feval_history(i,4));
    end
    if mod(i,1000)==0
         fprintf('\n');
    end    
end
fprintf('])\n\n'); 