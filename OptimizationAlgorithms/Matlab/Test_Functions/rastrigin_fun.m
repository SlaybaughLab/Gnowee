function z = rastrigin_fun(x)
    z=10.*length(x)+sum(x.^2.-10.*cos(2.*pi.*x));