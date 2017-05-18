function z = mi_chemical_process(x)
    z=(x(4)-1)^2 + (x(5)-2)^2 + (x(6)-1)^2 - log(x(7)+1) + (x(1)-1)^2 + (x(2)-2)^2 + (x(3)-3)^2;
    z=z+getnonlinear(x);

function Z=getnonlinear(x)
    Z=0;
    % Penalty constant
    lam=10^15;
    
    % Inequality constraints
    g(1)=x(1)+x(2)+x(3)+x(4)+x(5)+x(6)-5;
    g(2)=x(1)^2+x(2)^2+x(3)^2+x(6)^2-5.5;
    g(3)=x(1)+x(4)-1.2;
    g(4)=x(2)+x(5)-1.8;
    g(5)=x(3)+x(6)-2.5;
    g(6)=x(1)+x(7)-1.2;
    g(7)=x(2)^2+x(5)^2-1.64;
    g(8)=x(3)^2+x(6)^2-4.25;
    g(9)=x(3)^2+x(5)^2-4.64;

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