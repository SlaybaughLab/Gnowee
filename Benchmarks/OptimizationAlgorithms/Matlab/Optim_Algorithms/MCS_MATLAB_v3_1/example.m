
%   Example script for running MCS

%   Written by Sean Walton (512465@swansea.ac.uk) 2011 for Swansea
%   university

%   Please cite the following paper when using this code...
%   S.Walton, O.Hassan, K.Morgan and M.R.Brown "Modified cuckoo search: A
%   new gradient free optimisation algorithm" Chaos, Solitons & Fractals Vol
%   44 Issue 9, Sept 2011 pp. 710-718 DOI:10.1016/j.chaos.2011.06.004


%   I'd appreciate it if you contacted me (512465@swansea.ac.uk) if you apply the code to a
%   problem succesfully, I'm always interested in hearing about new applications 

clear all
clc

%The structure S contains all the parameters for the MCS
S.pa = 0.7;
S.plot = 1;         %If you want the results plotted set this to 1
S.constrain = 1;    %Set to 1 if you want the search constrained within vardef, zero otherwise
S.A = 100; %   Maximum distance a cuckoo can travel in one step as fraction of search space diagonal
S.pwr = 0.5;
S.flight = 1;
S.NesD = 1;
S.fname = 'obj';

%The matrix vardef defines the upper and lower bounds of the initial set of
%nests, the MCS uses this to set boundaries on the plots and LHC uses it to
%generate initial eggs

% NoDim = 10;
%vardef(1,1:NoDim) = 100;
%vardef(2,1:NoDim) = -100;

NoDim = 3;
vardef(1,1:3) = [2.0 1.3 15.0];
vardef(2,1:3) = [0.05 0.25 2.0];

NoNests = 100;

NestI = LHC(vardef,NoNests); %Generates initial set of eggs

NoGen = 100;

%Run optimiser
[p,F,pg,numEval,diversity] = ACuckoov3(NoGen, NestI, S, vardef);

%The optimum position is then pg
obj(pg)
