function z = mi_pressure_vessel(x)
    t=([0.0625 0.125 0.182 0.25 0.3125 0.375 0.4375 0.5 0.5625 0.625 0.6875 0.75 0.7125 0.875 0.9375 ...
        1 1.0625 1.125 1.1875 1.25 1.3125 1.375 1.4375 1.5 1.5625 1.625 1.6875 1.75 1.8125 1.875 1.9375 2]);
    z=0.6224*x(1)*t(x(3))*x(2)+1.7781*x(1)^2*t(x(4))+3.1611*t(x(3))^2*x(2)+19.8621*x(1)*t(x(3))^2;
    z=z+getnonlinear(x(1),x(2),t(x(3)),t(x(4)));

function Z=getnonlinear(R,L,ts,th)
    Z=0;
    % Penalty constant
    lam=10^15;
    
    % Inequality constraints
    g(1)=-ts+0.01932*R;
    g(2)=-th+0.00954*R;
    g(3)=-pi*R^2*L-4/3*pi*R^3+750*1728;
    g(4)=-240+L;

    % No equality constraint in this problem, so empty;
    geq=[];

    % Apply inequality constraints
    for k=1:length(g),
        Z=Z+lam*g(k)^2*getH(g(k));
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