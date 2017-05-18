function z = spring(x)
    d=([0.009 0.0095 0.104 0.0118 0.0128 0.0132 0.014 0.015 0.0162 0.0173 0.018 0.020 0.023 0.025 0.028 0.032 0.035 0.041 ...
           0.047 0.054 0.063 0.072 0.080 0.092 0.105 0.120 0.135 0.148 0.162 0.177 0.192 0.207 0.225 0.244 0.263 0.283 0.307 0.331 ...
           0.362 0.394 0.4375 0.500]);
    z=pi^2*x(1)*d(x(3))^2*(x(2)+2)/4;
    z=z+getnonlinear(x(2),x(1),d(x(3)));

function Z=getnonlinear(N,D,d)
    Z=0;
    % Penalty constant
    lam=10^15;
    
    % Variable Definititions:
    Fmax=1000;
    S=189000.0;
    Fp=300;
    sigmapm=6.0;
    sigmaw=1.25;
    G=11.5*10^6;
    lmax=14;
    dmin=0.2;
    Dmax=3.0;
    K=G*d^4/(8*N*D^3);
    sigmap=Fp/K;
    Cf=(4*(D/d)-1)/(4*(D/d)-4)+0.615*d/D;
    lf=Fmax/K+1.05*(N+2)*d;
    
    % Inequality constraints
    g(1)=8*Cf*Fmax*D/(pi*d^3)-S;
    g(2)=lf-lmax;
    g(3)=dmin-d;
    g(4)=D-Dmax;
    g(5)=3.0-D/d;
    g(6)=sigmap-sigmapm;
    g(7)=sigmap+(Fmax-Fp)/K + 1.05*(N+2)*d-lf;
    g(8)=sigmaw-(Fmax-Fp)/K;

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