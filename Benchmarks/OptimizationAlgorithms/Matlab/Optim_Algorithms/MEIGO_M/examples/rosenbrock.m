function [f, g]= rosenbrock(x)
f=0;
for i=1:length(x)-1
    f=f+((x(i)-1)^2+100.*(x(i+1)-x(i)^2)^2);
end
g=[];
return