function Results = CeSS(par_struct,n_iter,is_parallel)

	n_threads=length(par_struct);
	
	Results=[];
	f=[];
	x=[];
	temp_f=[];
	Results.results_iter={};
	
	numeval=0;
	neval=0;
	time=0;

	Nrunsmat(1,1,1:n_threads) = 1:n_threads; 

	for iteration = 1:n_iter   
		
		tstart = tic;		
		results_iter=[];
		
		%With JPAR:
		if(is_parallel)		
			results_iter = jpar_client('CeSS_thread', par_struct, Nrunsmat);
		%Without JPAR (useful for debugging):	
		else
		
			for thread=1:n_threads
			
				res = ess_kernel(par_struct(thread).problem, par_struct(thread).opts);
				results_iter(thread).x=res.x;
				results_iter(thread).f=res.f;
				results_iter(thread).time=res.time;
				results_iter(thread).refset_x=res.Refset.x;
				results_iter(thread).refset_f=res.Refset.f;
				results_iter(thread).neval=res.neval;
				results_iter(thread).numeval=res.numeval;
				results_iter(thread).cpu_time=res.cpu_time;
				results_iter(thread).fbest=res.fbest;
				results_iter(thread).xbest=res.xbest;
				
			end
			
		end
		
		%Store the results from all threads in this iteration
		Results.results_iter{iteration}=results_iter;
		
		x_0=[];
		f_0=[];
		
		%concatenate all refsets and best solutions found
		for thread=1:n_threads
			x_0=[x_0;results_iter(thread).refset_x;results_iter(thread).xbest];
			f_0=[f_0;results_iter(thread).refset_f;results_iter(thread).fbest];
			if(iteration==1)
				temp_f=[temp_f results_iter(thread).f(1)];
			end
			numeval=numeval+results_iter(thread).numeval;
		end
		
		%Measure the duration of the iteration
		time=[time toc(tstart)];
		%Get an initial solution
		if(iteration==1)
			f=[f min(temp_f)];
			fi=find(min(temp_f));
			x=[x;results_iter(fi(1)).x(1,:)];
		end
		
		%Remove repeated elements from x_0 and f_0
		[C ia ic]=unique(x_0,'rows');
		x_0=x_0(sort(ia),:);
		f_0=f_0(sort(ia));
		
		%Assign all refsets as initial solution
		for thread=1:n_threads
			par_struct(thread).problem.x_0=x_0;
			par_struct(thread).problem.f_0=f_0';
		end
		
		fbest=min(f_0);
		xbest=x_0(find(f_0==fbest),:);
		
		f=[f; fbest(1)];
		x=[x; xbest(1,:)];
		neval=[neval; numeval];
		
    end
    

	Results.xbest=xbest;
	Results.fbest=fbest;
    Results.f_iter=f;
	Results.x_iter=x;
    Results.time_iter=time;
    
    total_vals=[0; f(1)];
    %Put all the results together
    for j=1:n_iter
        t_vals = [];
        f_vals = [];
        for k = 1:n_threads
            
            t_vals = [t_vals time(j)+Results.results_iter{j}(k).time];
            f_vals = [f_vals Results.results_iter{j}(k).f];
        end
        total_vals_thread = [t_vals; f_vals];
        total_vals = [total_vals total_vals_thread];
       
    end
    
    [Times,I] = sort(total_vals(1,:));
    Values    = total_vals(2,I);
    Final     = [Times(1);Values(1)];
    for m = 2:size(Values,2)    
        if Values(m) <= Final(2,size(Final,2))
            Final = [Final [Times(m); Values(m)] ];
        end    
    end    
    

	Results.time=Final(1,:);
    Results.f=Final(2,:);
	Results.neval=neval;
	Results.numeval=numeval;

end