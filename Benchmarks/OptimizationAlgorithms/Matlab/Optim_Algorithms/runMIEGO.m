clear all
clc


%========================= SPRING PROBLEM SPECIFICATIONS ===========================
% problem.f='spring_fun';
% nvar=3;
% problem.x_L=([0.05 0.25 2.0]);
% problem.x_U=([2.0 1.3 15.0]);
% problem.int_var=0;
% optim=0.012665;

%========================= MI SPRING PROBLEM SPECIFICATIONS ===========================
problem.f='mi_spring';
nvar=3;
problem.x_L=([0.01 1 1]);
problem.x_U=([3.0 10 42]);
problem.int_var=2;
% global optimum
%x*=[1.22311, 9, 36];
optim=2.65872;

%========================= PRESSURE VESSEL PROBLEM SPECIFICATIONS ===========================
% problem.f='pressureVessel';
% nvar=4;
% % [R,L,t_s,t_h]
% problem.x_L=([25.0 25.0 18 10]);
% problem.x_U=([210.0 240.0 32 32]);
% problem.int_var=2;
% % global optimum
% %x*=[58.2298 44.0291 18 10];
% optim=5855.893191;

%========================= CHEMICAL PROCESS PROBLEM SPECIFICATIONS ===========================
% problem.f='minlp_chemical_process_fun';
% nvar=7;
% % [x1,x2,x3,y1,y2,y3,y4]
% problem.x_L=([0,0,0,0,0,0,0]);
% problem.x_U=([10.0,10.0,10.0,1,1,1,1]);
% problem.int_var=4;
% % global optimum
% %x*=[0.2, 0.8, 1.907878, 1, 1, 0, 1];
% optim=4.579582;

%========================= BENCHMARK SPECIFICATIONS ===========================
iter=10;
opts.maxeval=200000;
opts.ndiverse=50; 
opts.n_stuck=250;
conTol=1E-2;

%========================= SOLVER SPECIFICATIONS ===========================
opts.local.solver=0;
algorithm='eSS';
opts.inter_save=1;

% Initialize variables
fitness=[];
feval=[];
design=[];
bestf=1E15;

% Run Optimization
for i=1:iter
    [Results]=MEIGO(problem,opts,algorithm);
    
    for j=1:length(Results.f) 
       if Results.f(j) <= optim*(1.0+conTol)
           break
       end
    end
    fitness(i)=Results.f(j);
    feval(i)=Results.neval(j);
    design(i,:)=Results.x(end,:);
    
    % Save best design information
    if i==1 | fitness(i)<bestf
        bestdesign=design(i,:);
        bestf=fitness(i);
        beste=feval(i);
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
% 
%    % Plot Histogram
%    figure(2);
%    h=histogram(feval,100,'Normalization','probability');
%    set(h,'FaceColor','k','EdgeColor','k');
%    xlabel('# Function Evals')
%    ylabel('Probability')
%    title('Function Evaluations for Welded Beam Optimization using GA')
%       
%    
%    % Calculate % Diff, standard deviation, and output feval history for use in plotting in python
%    fprintf('np.array([');
%    for i=1:length(feval_history)
%        % Calculate standard deviation
%        if feval_history(i,3)~=0
%            feval_history(i,3)=sqrt(feval_history(i,3)./(feval_history(i,4)-1))*100;
%        end
%        % Calculate % diff
%        if optim==0.0
%            feval_history(i,2)=abs(feval_history(i,2))*100;
%        else
%            feval_history(i,2)=abs((feval_history(i,2)-optim))/optim*100;
%        end
%        % Output data 
%        if i==length(feval_history)
%             fprintf('[%d,%f,%f, %d]', feval_history(i,1),feval_history(i,2),feval_history(i,3),feval_history(i,4));
%        else
%             fprintf('[%d,%f,%f, %d],', feval_history(i,1),feval_history(i,2), feval_history(i,3), feval_history(i,4));
%        end
%        if mod(i,1000)==0
%             fprintf('\n');
%        end    
%    end
%    fprintf('])\n\n'); 