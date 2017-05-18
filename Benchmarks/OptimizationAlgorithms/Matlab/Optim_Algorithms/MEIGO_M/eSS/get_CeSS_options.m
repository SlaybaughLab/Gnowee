function par_struct = get_CeSS_options(n_threads, npars, maxtime_per_iteration)
	
	problem=[];
	%--- Problem definition: --------------------------------------------------
	problem.x_L = [];
	problem.x_U = [];
	problem.f  = '';

	%--- Optimization settings, different for each thread: --------------------
	dim1 = 0.5;   bal1 = 0;     n2_1 = 0;
	dim2 = 1;     bal2 = 0;     n2_2 = 0;
	dim3 = 2;     bal3 = 0;     n2_3 = 2;
	dim4 = 3;     bal4 = 0;     n2_4 = 4;
	dim5 = 5;     bal5 = 0.25;  n2_5 = 7;
	dim6 = 5;     bal6 = 0.25;  n2_6 = 10;
	dim7 = 7.5;   bal7 = 0.25;  n2_7 = 15;
	dim8 = 10;    bal8 = 0.5;   n2_8 = 20;
	dim9 = 12;    bal9 = 0.25;  n2_9 = 50;
	dim10 = 15;   bal10 = 0.25; n2_10 = 100;

	dim = [dim1 dim2 dim3 dim4 dim5 dim6 dim7 dim8 dim9 dim10];
	bal = [bal1 bal2 bal3 bal4 bal5 bal6 bal7 bal8 bal9 bal10];
	n2  = [n2_1 n2_2 n2_3 n2_4 n2_5 n2_6 n2_7 n2_8 n2_9 n2_10];

	opts.maxtime       = maxtime_per_iteration;
	opts.local.solver  = 'dhc'; 
	opts.local.finish  = 'dhc';
	opts.local.tol     = 2;        
	opts.local.iterprint = 0;

	i=0;
	counter=0;
	while(i<n_threads)
		i=i+1;
		counter=counter+1;
		par_struct(i).opts=opts;
		par_struct(i).problem=problem;
		nnn = roots([1  -1 -dim(counter)*npars]);
		iii = find(nnn>0);
		par_struct(i).opts.dim_refset    = ceil(nnn(iii));
		par_struct(i).opts.local.balance = bal(counter);
		par_struct(i).opts.local.n2      = n2(counter);
		par_struct(i).opts.maxeval=50;
		%If there are more than 10 threads or is a multiple of 10
		%start recycling options or restart counter
		if(mod(counter,10)==0)
			counter=0;
		end
	end
	
	return;
	
end