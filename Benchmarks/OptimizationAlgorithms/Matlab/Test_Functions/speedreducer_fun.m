function z = speedreducer_fun(x)
    z=0.7854*x(1)*x(2)^2*(3.3333*x(3)^2+14.9334*x(3)-43.0934) ...
            - 1.508*x(1)*(x(6)^2+x(7)^2) + 7.4777*(x(6)^3+x(7)^3) ...
            + 0.7854*(x(4)*x(6)^2+x(5)*x(7)^2);
    z=z+getnonlinear(x);

function Z=getnonlinear(x)
    Z=0;
    % Penalty constant
    lam=10^20;

    % Inequality constraints
    g(1)=(27./(x(1)*x(2)^2*x(3))-1.);
    g(2)=(397.5/(x(1)*x(2)^2*x(3)^2)-1.);
    g(3)=(1.93*x(4)^3/(x(2)*x(3)*x(6)^4)-1.);
    g(4)=(1.93*x(5)^3/(x(2)*x(3)*x(7)^4)-1.);
    g(5)=(1.0/(110.*x(6)^3)*sqrt((745.0*x(4)/(x(2)*x(3)))^2+16.9*10^6)-1);
    g(6)=(1.0/(85.*x(7)^3)*sqrt((745.0*x(5)/(x(2)*x(3)))^2+157.5*10^6)-1);
    g(7)=(x(2)*x(3)/40.-1);
    g(8)=(5.*x(2)/x(1)-1);
    g(9)=(x(1)/(12.*x(2))-1);
    g(10)=((1.5*x(6)+1.9)/x(4)-1);
    g(11)=((1.1*x(7)+1.9)/x(5)-1);

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