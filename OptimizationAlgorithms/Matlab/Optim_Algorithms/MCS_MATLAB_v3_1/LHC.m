function [samples] = LHC(vardef,N)
%LHC Produces latin hypercube sampling of variables defines by vardef
%   vardef(2,NoVar) - first row gives max value second min value
%   N = number of samples

% Written by Sean Walton 2011 for Swansea university

%Definitions

[dummy,NoVar] = size(vardef);

diff = zeros(1,NoVar);
samples = zeros(N,NoVar);


%Find difference for scaling
for i=1:NoVar
    diff(1,i) = (vardef(2,i) - vardef(1,i));
end

norm_samples = lhsdesign(N,NoVar,'smooth','off','criterion','maximin','iterations',100);

%Scale
for i=1:N
    for j=1:NoVar
        samples(i,j) = vardef(1,j) + diff(1,j)*norm_samples(i,j);
    end
end
end

