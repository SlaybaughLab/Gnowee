function A = ackley(x)
% ackley(x); x: vector, real
c1 = 20;
c2 = 0.2;
c3 = 2*pi;
A = -c1 * exp(-c2*sqrt(mean(x.^2))) - exp(mean(cos(c3*x))) + c1 + exp(1);
end

% To plot the Ackley function in 3D, type:
% func = @(x) ackley(20, 0.2, 2*pi, x)
% plot_2d_function(func, -5 * ones(2,1), 5 * ones(2,1))