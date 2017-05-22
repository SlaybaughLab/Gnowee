clc
clear all

fun = @mi_pressure_vessel;
   nvars = 4;
   optim=6059.714335;
   lb = ([10.0 1E-8 1 1]);
   ub = ([50.0 200.0 99 99 ]);
   intVar=[3,4];

% fun = @mi_spring;
%    nvars = 3;
%    optim=2.65856;
%    lb = ([0.0 1 1]);
%    ub = ([3.0 10 42]);   
%    intVar=[2,3];
% 
%  fun = @mi_chemical_process;
%     nvars=7;
%     optim=4.579582;
% 	% [x1,x2,x3,y1,y2,y3,y4]
% 	lb=([0,0,0,0,0,0,0]);
% 	ub=([10.0,10.0,10.0,1,1,1,1]);
%     intVar=[4,5,6,7];

   
   %rng default; 
   
   % Initialize Variables
   n=100;   %number of iterations
   design=[[]];
   fitness=[];
   feval=[];
   nonconverge=0;
   bestf=1E15;
   hist_interval=500;
   feval_history = zeros(2,4);
   if optim == 0.0
       opLim=0.01;
   else
       opLim=optim*1.01;
   end
   
   % Set GA Options and outputs   
%    options = gaoptimset('PopulationSize',50,'Generations',4000, ...
%                           'StallGenLimit',200,'TolFun',1E-6,'TolCon',1E-6,'FitnessLimit',opLim);
   options = gaoptimset('PopulationSize',50,'Generations',200, ...
                          'StallGenLimit',200,'TolFun',1E-12,'TolCon',1E-12);
   %'PlotFcns':@gaplotstopping,@gaplotexpectation,@gaplotscorediversity,@gaplotbestindiv,@gaplotbestf
   
   % Run GA n times
   for i=1:n
       [x,fval,exitflag,output] = ga(fun,nvars,[],[],[],[],lb,ub,[],intVar,options);
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
   title('Function Evaluations for Welded Beam Optimization using GA')
      
   
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