function Results = CeSS_thread(par_struct,Nruns)
		
		%Reset the random seed.
		randstate = 1e3*Nruns + sum(100*clock);
		rand('state',randstate);
		randn('state',randstate);

		%Run optimization
		res = ess_kernel(par_struct(Nruns).problem, par_struct(Nruns).opts);
		
		%Store results
		Results.x=res.x;
		Results.f=res.f;
		Results.refset_x=res.Refset.x;
		Results.refset_f=res.Refset.f;
		Results.neval=res.neval;
		Results.numeval=res.numeval;
		Results.cpu_time=res.cpu_time;
		Results.fbest=res.fbest;
		Results.xbest=res.xbest;
		Results.time=res.time;
		
end