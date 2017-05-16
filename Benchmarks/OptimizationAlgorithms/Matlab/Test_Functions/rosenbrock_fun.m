function z = rosenbrock_fun(x)
    z=0;
    for i=1:length(x)-1
        z=z+((x(i)-1)^2+100.*(x(i+1)-x(i)^2)^2);
    end