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

function  net = cln_create_som(opt, din)
% create struct for SOM
net(1:opt.net.size, 1:opt.net.size) = struct('xpos', 0,...
    'ypos', 0,...
    'W'   , zeros(1, opt.data.trainvsize),...          % input weights
    'H'   , zeros(opt.net.size, opt.net.size),...  % Hebbian weights for cross-SOM interaction
    'ad'  , 0.0, ... % direct activation elicited by input vector
    'ai'  , 0.0, ... % indirect activation elicited by cross-SOM interaction
    'at'  , 0.0 ...  % total joint activation (direct + indirect)
    );

% in order to speed-up computation we init weights in the bounds of the
% input vectors interval
minin = min(din); maxin = max(din);

% initialize SOM
for idx = 1:opt.net.size
    for jdx = 1:opt.net.size
        net(idx, jdx).xpos = idx;
        net(idx, jdx).ypos = jdx;
        for in_idx = 1:opt.data.trainvsize
            net(idx, jdx).W(in_idx) = minin + (maxin - minin)*rand;
        end
        for kidx = 1:opt.net.size
            for tidx = 1:opt.net.size
                net(idx, jdx).H(kidx, tidx)  = minin + (maxin - minin)*rand;
            end
        end
%         net(idx, jdx).ad = minin + (maxin - minin)*rand;
%         net(idx, jdx).ai = minin + (maxin - minin)*rand;
%         net(idx, jdx).at = minin + (maxin - minin)*rand;
    end
end

end