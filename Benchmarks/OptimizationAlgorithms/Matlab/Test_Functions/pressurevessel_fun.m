function z = pressurevessel_fun(x)
    z=0.6224*x(1)*x(3)*x(4)+1.7781*x(2)*x(3)^2+3.1661*x(1)^2*x(4)+19.84*x(1)^2*x(3);
    z=z+getnonlinear(x);

function Z=getnonlinear(x)
    Z=0;
    % Penalty constant
    lam=10^20;

    % Inequality constraints
    g(1)=(-x(1)+0.0193*x(3));
    g(2)=(-x(2)+0.00954*x(3));
    g(3)=(-pi*x(3)^2*x(4)-4./3.*pi*x(3)^3+1296000);
    g(4)=(x(4)-240);
    
    % No equality constraint in this problem, so empty;
    geq=[];

    % Apply inequality constraints
    for k=1:length(g),
        Z=Z+ lam*g(k)^2*getH(g(k));
    end
    % Apply equality constraints
    for k=1:length(geq),
       Z=Z+lam*geq(k)^2*getHeq(geq(k));
end

% Test if inequalities hold
% Index function H(g) for inequalities
function H=getH(g)
    if g<=0,
        H=0;
    else
        H=1;
end
% Index function for equalities
function H=getHeq(geq)
    if geq==0,
       H=0;
    else
       H=1;
end