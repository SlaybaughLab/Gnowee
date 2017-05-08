function z = yang(N) 
beta=3/2;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

u=randn(1,N)*sigma;
v=randn(1,N);
z=u./abs(v).^(1/beta);