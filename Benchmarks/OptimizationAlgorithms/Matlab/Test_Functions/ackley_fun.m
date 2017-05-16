function z = ackley_fun(x)
    z=-20*exp(-1./5.*sqrt(1./length(x)*sum(x.^2))) - exp(1./length(x)*sum(cos(2.*pi.*x))) + 20 + exp(1);