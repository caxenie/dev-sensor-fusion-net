%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% For the current implementation we only consider a 2 variable network

% FUNCTION
% Extracts and prepares net runtime data for visualization.
%
% ARGS
%   opts - simulation options parametrized for analysis
%
% RETURN
%   data - struct with data to be fed into visualization module

function visin = cln_runtime_dataset_setup(opts)
    runtime_data = load(opts.data.infile);
    visin = runtime_data;
end