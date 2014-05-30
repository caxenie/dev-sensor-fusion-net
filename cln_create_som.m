%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% For the current implementation we only consider a 2 variable network

% FUNCTION
% Creates a SOM given input parameters
%
% ARGS
%   opt - simulation options parametrized for run
%   din - associated sensory input variable vector
%
% RETURN
%   net - struct with data need to run the global network dynamics

function  net = cln_create_som(opt, din, idn)
% create struct for SOM
net(1:opt.net.sizex, 1:opt.net.sizey) = struct('xpos', 0,...
    'ypos', 0,...
    'W'   , zeros(1, opt.data.trainvsize),...          % input weights
    'H'   , zeros(opt.net.sizex, opt.net.sizey),...  % Hebbian weights for cross-SOM interaction
    'ad'  , 0.0, ... % direct activation elicited by input vector
    'ai'  , 0.0, ... % indirect activation elicited by cross-SOM interaction
    'at'  , 0.0, ... % total joint activation (direct + indirect)
    'id'  , idn ...   % id of the network  
    );

% in order to speed-up computation we init weights in the bounds of the
% input vectors interval
minin = min(din); maxin = max(din);

% initialize SOM
for idx = 1:opt.net.sizex
    for jdx = 1:opt.net.sizey
        net(idx, jdx).xpos = idx;
        net(idx, jdx).ypos = jdx;
        for in_idx = 1:opt.data.trainvsize
            if(strcmp(idx, 'som1')==1)
                net(idx, jdx).W(in_idx) = minin + (maxin - minin)*rand;
            else
                net(idx, jdx).W(in_idx) = minin + (maxin - minin)*rand;
            end
        end
        for kidx = 1:opt.net.sizex
            for tidx = 1:opt.net.sizey
                net(idx, jdx).H(kidx, tidx)  = rand;
            end
        end
    end
end
fprintf(1, 'cln_create_som: Created %s of size %d x %d\n', idn, opt.net.sizex, opt.net.sizey);
end