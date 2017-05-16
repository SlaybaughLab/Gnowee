clc;
clear all;

fileID=fopen('C:\Users\James\Dropbox\UCB\Research\ETAs\Design\Matlab\Test_Functions\Ch150.tsp');
optim=6528.0;

% Remove header lines
for i=1:6
    tline = fgetl(fileID);
end

%Store city pairs
tmp=textscan(fileID,'%f %f %f');
tmp=cell2mat(cellfun(@(x)x(:,:),tmp,'un',0));
xy_orig=tmp(:,2:3);
fclose(fileID);

%Initialize variables
n=50;  %Number of times to run algorithm
numIter=2000;
stallLim=150;
fitness=[];
feval=[];
nonconverge=0;
bestf=1E15;
hist_interval=500;
feval_history = zeros(2,4);

% Run GA TSP algorithm
for i=1:n
   xy=xy_orig;
%   userConfig = struct('xy',xy,'popSize',100,'numIter',2000, 'stallLimit', 150, 'convTol', 1E-2, 'showwaitbar',true);
   userConfig = struct('xy',xy,'popSize',100,'numIter',numIter, 'stallLimit', stallLim, 'convTol', 1E-2, 'optimim', optim, 'showprog', false, 'showresult', false, 'showwaitbar',false);
   resultStruct = tsp_ga(userConfig);
   
   % Save run results
   fitness(i)=resultStruct.minDist;
   feval(i)=max(resultStruct.fevalHist);
   if resultStruct.minDist<=bestf
       bestf=resultStruct.minDist;
       beste=feval(i);
   end 
   
   % Save function eval history results to a predefined binned
   % structure; Compute the average and standard deviation using a recurrence relation
   k=2;
   for j=1:length(resultStruct.fevalHist)
       if j > length(feval_history)
           feval_history = [feval_history; [0,0.0,0.0,0]];
       end
       feval_history(j,1)=500*(j-1);
       while resultStruct.fevalHist(k) < feval_history(j,1) && k+1<=length(resultStruct.fevalHist) 
            k=k+1;
       end
       if k+1 > length(resultStruct.fevalHist) && resultStruct.fevalHist(k) < feval_history(j,1)
            % Initialize the array on the first run
            if feval_history(j,4)==0
                feval_history(j,2)=resultStruct.distHist(k);
                feval_history(j,3)=0.0;
                feval_history(j,4)=feval_history(j,4)+1;
            else
                old_mean=feval_history(j,2);
                feval_history(j,4)=feval_history(j,4)+1;
                feval_history(j,2)=feval_history(j,2)+(resultStruct.distHist(k)-feval_history(j,2))/feval_history(j,4);
                feval_history(j,3)=feval_history(j,3)+(resultStruct.distHist(k)-old_mean)*(resultStruct.distHist(k)-feval_history(j,2));
            end
            break 
       else
            % Initialize the array on the first run
            if feval_history(j,4)==0
                feval_history(j,2)=resultStruct.distHist(k);
                feval_history(j,3)=0.0;
                feval_history(j,4)=feval_history(j,4)+1;
            else
                old_mean=feval_history(j,2);
                feval_history(j,4)=feval_history(j,4)+1;
                feval_history(j,2)=feval_history(j,2)+(resultStruct.distHist(k-1)-feval_history(j,2))/feval_history(j,4);
                feval_history(j,3)=feval_history(j,3)+(resultStruct.distHist(k-1)-old_mean)*(resultStruct.distHist(k-1)-feval_history(j,2));
            end
       end
   end
end

% Output Average Solutions
   fprintf('\nThe Average Optimized Solution:\n')
   fprintf('------------------------------------\n')
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
   title('Function Evaluations for TSP Optimization using GA')
   
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
       if mod(i,500)==0
            fprintf('\\\n');
       end    
   end
   fprintf('])\n\n');  
