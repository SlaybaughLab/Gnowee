function z = mi_pressure_vessel(x)
    t=linspace(0.0625, 6.1875, 99);
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