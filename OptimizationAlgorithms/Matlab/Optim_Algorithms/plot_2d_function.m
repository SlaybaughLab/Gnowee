function plot_2d_function(func, lb, ub)
% plot_2d_function(func, lb, ub); func: @f(x); lb, ub: vector, real
[X,Y] = meshgrid(lb(1):.1:ub(1), lb(2):.1:ub(2));
for (i=1:1:length(X(:,1)))
for (j=1:1:length(X(:,1)))
Z(i,j) = func([X(i,j); Y(i,j)]);
end
end
surfc(X,Y,Z)