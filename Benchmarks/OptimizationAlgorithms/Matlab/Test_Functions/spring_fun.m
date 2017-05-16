function z = my_fun(x)
    z=(2+x(3))*x(1)^2*x(2);
    z=z+getnonlinear(x);

function Z=getnonlinear(u)
    Z=0;
    % Penalty constant
    lam=10^20;

    % Inequality constraints
    g(1)=1-u(2)^3*u(3)/(71785*u(1)^4);
    gtmp=(4*u(2)^2-u(1)*u(2))/(12566*(u(2)*u(1)^3-u(1)^4));
    g(2)=gtmp+1/(5108*u(1)^2)-1;
    g(3)=1-140.45*u(1)/(u(2)^2*u(3));
    g(4)=(u(1)+u(2))/1.5-1;

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