function z = griewank_fun(x)
    tmp=1;
    for i=1:length(x)
        tmp=tmp*cos(x(i)/sqrt(i));
    end
    z=1./4000.*sum(x.^2) - tmp +1.;