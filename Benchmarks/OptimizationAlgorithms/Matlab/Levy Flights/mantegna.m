function z = mantegna(alpha, gam, n, N) 
% Errortraps: 
if (alpha < 0.3 | alpha > 1.99) 
    disp('Valid range for alpha is [0.3;1.99].') 
    z = NaN * zeros(1,N); 
    return 
end 
if (gam <= 0) 
    disp('c must be positive.') 
    z = NaN * zeros(1,N); 
    return 
end 
if (n < 1) 
    disp('n must be positive.') 
    z = NaN * zeros(1,N); 
    return 
end 
if nargin<4 
    N=1; 
end 
if (N <= 0) 
    disp('N must be positive.')      
    z = NaN; 
    return 
end

invalpha = 1/alpha;
sigx = ((gamma(1+alpha)*sin(pi*alpha/2))/(gamma((1+alpha)/2)... 
       *alpha*2^((alpha-1)/2)))^invalpha; 
v = sigx*randn(n,N)./(abs(randn(n,N)).^invalpha); 
kappa = (alpha*gamma((alpha+1)/(2*alpha)))/gamma(invalpha)... 
        *((alpha*gamma((alpha+1)/2))/(gamma(1+alpha)*sin(pi*alpha/2)))^invalpha; 
p =  [-17.7767  113.3855 -281.5879  337.5439 -193.5494   44.8754]; 
c = polyval(p, alpha); 
%c=2.737
w = ((kappa-1)*exp(-abs(v)./c)+1).*v; 
if(n>1) 
    z = (1/n^invalpha)*sum(w); 
else 
	z = w; 
end
z = gam^invalpha*z;