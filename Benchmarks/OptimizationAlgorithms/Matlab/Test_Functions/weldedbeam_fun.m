function z = weldedbeam_fun(x)
    z=1.10471*x(1)^2*x(2)+0.04811*x(3)*x(4)*(14.0+x(2));
    z=z+getnonlinear(x);

function Z=getnonlinear(x)
    Z=0;
    % Penalty constant
    lam=10^20;

    % Inequality constraints
    em=6000.*(14+x(2)/2.);
    r=sqrt(x(2)^2/4.+((x(1)+x(3))/2.)^2);
    j=2.*(x(1)*x(2)*sqrt(2)*(x(2)^2/12.+((x(1)+x(3))/2.)^2));
    tau_p=6000./(sqrt(2)*x(1)*x(2));
    tau_dp=em*r/j;
    tau=sqrt(tau_p^2+2.*tau_p*tau_dp*x(2)/(2.*r)+tau_dp^2);
    sigma=504000./(x(4)*x(3)^2);
    delta=65856000./((30*10^6)*x(4)*x(3)^2);
    pc=4.013*(30.*10^6)*sqrt(x(3)^2*x(4)^6/36.)/196.*(1.-x(3)*sqrt((30.*10^6)/(4.*(12.*10^6)))/28.);
    g(1)=(tau-13600.);
    g(2)=(sigma-30000.);
    g(3)=(x(1)-x(4));
    g(4)=(0.10471*x(1)^2+0.04811*x(3)*x(4)*(14.+x(2))-5.0);
    g(5)=(0.125-x(1));
    g(6)=(delta-0.25);
    g(7)=(6000-pc);

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