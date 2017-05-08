clc
clear all

% fun = @weldedbeam_fun;
%    nvars = 4;
%    optim=1.724852;
%    lb = ([0.1 0.1 1E-8 1E-8]);
%    ub = ([10.0 10.0 10.0 2.0]);
%    
% fun = @pressurevessel_fun;
%    nvars = 4;
%    optim=6059.714335;
%    optim=5885.33285347;
%    lb = ([0.0625 0.0625 10.0 1E-8]);
%    ub = ([1.25 99*0.0625 50.0 200.0]);
%    
% fun = @speedreducer_fun;
%    nvars = 7;
%    optim=2996.348165;
%    lb = ([2.6 0.7 17.0 7.3 7.8 2.9 5.0]);
%    ub = ([3.6 0.8 28.0 8.3 8.3 3.9 5.5]);
%
% fun = @spring_fun;
%    nvars = 3;
%    optim=0.012665;
%    lb = ([0.05 0.25 2.0]);
%    ub = ([2.0 1.3 15.0]);   
%       
fun = @ackley_fun;
   nvars = 3;
   optim=0.0;
   lb = ([-25.0 -25.0 -25.0]);
   ub = ([25.0 25.0 25.0]);
%    
% fun = @dejong_fun;
%    nvars = 4;
%    optim=0.0;
%    lb = ([-5.12 -5.12 -5.12 -5.12]);
%    ub = ([5.12 5.12 5.12 5.12]);
%     
% fun = @easom_fun;
%    nvars = 2;
%    optim=-1.0;
%    lb = ([-100.0 -100.0]);
%    ub = ([100.0 100.0]);
%    
% fun = @griewank_fun; 
%    nvars = 6;
%    optim=0.0;
%    lb = ([-600.0 -600.0 -600.0 -600.0 -600.0 -600.0]);
%    ub = ([600.0 600.0 600.0 600.0 600.0 600.0]);
%    
% fun = @rastrigin_fun;
%    nvars = 5;
%    optim=0.0;
%    lb = ([-5.12 -5.12 -5.12 -5.12 -5.12]);
%    ub = ([5.12 5.12 5.12 5.12 5.12]);
%    
% fun = @rosenbrock_fun;
%    nvars = 5;
%    optim=0.0;
%    lb = ([-5.0 -5.0 -5.0 -5.0 -5.0]);
%    ub = ([5.0 5.0 5.0 5.0 5.0]);
   
   %rng default; 
   
   % Initialize Variables
   n=10;   %number of iterations
   design=[[]];
   fitness=[];
   feval=[];
   nonconverge=0;
   bestf=1E15;
   x0=abs(lb)+ub./4.;
   hist_interval=500;
   feval_history = zeros(2,4);
   if optim == 0.0
       opLim=0.01;
   else
       opLim=optim*1.01;
   end
   
   % Set SA Options and outputs
   % to get plots, add:'PlotFcns',@saplotbestf,
   options = saoptimset('MaxIter',200000,'StallIterLimit',10000, ...
                        'MaxFunEvals',200000,'TolFun',1E-6,'ObjectiveLimit',opLim);
%    options = saoptimset('MaxIter',10000,'StallIterLimit',10000, ...
%                         'MaxFunEvals',10000,'TolFun',1E-16);
                    
   % Run SA n times
   for i=1:n
       [x,fval,exitflag,output] = simulannealbnd(fun,x0,lb,ub,options);
       design(i,:)=x;
       fitness(i)=fval;
       feval(i)=output.funccount;
       if exitflag~=-3
           nonconverge=nonconverge+1;
       end
       if fval<=bestf
           bestf=fval;
           beste=feval(i);
           bestdesign=x;
       end
       
       % Save function eval history results to a predefined binned
       % structure; Compute the average and standard deviation using a recurrence relation
       k=2;
       for j=1:length(output.fevalhist)
           if j > length(feval_history)
               feval_history = [feval_history; [0,0.0,0.0,0]];
           end
           feval_history(j,1)=500*(j-1);
           while output.fevalhist(k) < feval_history(j,1) && k+1<=length(output.fevalhist) 
                k=k+1;
           end
           if k+1 > length(output.fevalhist) && output.fevalhist(k) < feval_history(j,1)
                % Initialize the array on the first run
                if feval_history(j,4)==0
                    feval_history(j,2)=output.fithist(k);
                    feval_history(j,3)=0.0;
                    feval_history(j,4)=feval_history(j,4)+1;
                else
                    old_mean=feval_history(j,2);
                    feval_history(j,4)=feval_history(j,4)+1;
                    feval_history(j,2)=feval_history(j,2)+(output.fithist(k)-feval_history(j,2))/feval_history(j,4);
                    feval_history(j,3)=feval_history(j,3)+(output.fithist(k)-old_mean)*(output.fithist(k)-feval_history(j,2));
                end
                break 
           else
                % Initialize the array on the first run
                if feval_history(j,4)==0
                    feval_history(j,2)=output.fithist(k);
                    feval_history(j,3)=0.0;
                    feval_history(j,4)=feval_history(j,4)+1;
                else
                    old_mean=feval_history(j,2);
                    feval_history(j,4)=feval_history(j,4)+1;
                    feval_history(j,2)=feval_history(j,2)+(output.fithist(k-1)-feval_history(j,2))/feval_history(j,4);
                    feval_history(j,3)=feval_history(j,3)+(output.fithist(k-1)-old_mean)*(output.fithist(k-1)-feval_history(j,2));
                end
           end
       end       
       
   end

   % Output Average Solutions
   fprintf('\nThe Average Optimized Solution:\n')
   fprintf('------------------------------------\n')
   fprintf('Design:\n')
   for i=1:length(ub)
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
   for i=1:length(ub)
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
   title('Function Evaluations for Welded Beam Optimization using SA')
      
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
            fprintf('[%d,%f,%f, %d]', feval_history(i,1),feval_history(i,2),feval_history(i,3),feval_history(i,4));
       else
            fprintf('[%d,%f,%f, %d],', feval_history(i,1),feval_history(i,2), feval_history(i,3), feval_history(i,4));
       end
       if mod(i,1000)==0
            fprintf('\n');
       end    
   end
   fprintf('])\n\n');     
   
   % Output feval history for use in plotting in python
   fprintf('np.array([');
   for i=1:length(feval)
       % Output data 
       if i==length(feval)
            fprintf('%d', feval(i));
       else
            fprintf('%d,', feval(i));
       end    
   end
   fprintf('])\n\n');  