%% UNSU PERVISED CORRELATION LEARNING NETWORK USING SOM
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
    try
        runtime_data = load(opts.data.infile);
    catch lasterr;
        fprintf('cln_runtime_dataset_setup: %s !\n', lasterr.message);
        visin = struct([]);
        return;
    end
    visin = runtime_data;
    fprintf(1, 'cln_runtime_dataset_setup: Visualization dataset was set up.\n');
end