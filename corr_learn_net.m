%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% Main simulation script
%% SETUP ENVIRONMENT

clear all; close all; clc
%% LOAD DATA AND SETUP RUNTIME

% simulation options parametrization
simopts.mode            = 'analyze';                    % mode given the function of the script, i.e. run, analyze
simopts.verbose         = 0;                        % flag to activate / inactivate debug verbose
simopts.execution       = 'iterative';              % mode to run the simulation: iterative / vectorial
simopts.data.source     = 'sensors';                % data source: generated or sensors (data from robot)
simopts.data.infile     = 'robot_data_jras_paper';  % robot dataset
simopts.data.scaling    = -0.0572957795130823;      % data dependent scaling
simopts.data.freqsamp   = 25;                       % Hz
simopts.data.trainvtype = 'full';                   % train vector type, i.e. fixed interval / sliding window / full dataset
simopts.data.trainvsize = 5062;                      % size (in samples) of the input vector
simopts.data.corrtype   = 'algebraic';              % input data correlation type, i.e. algebraic, temporal, nonlinear
% parametrize the network
simopts.net.size        = 10;                       % size x size square lattice SOM nets
simopts.net.alpha       = 0.1;                      % initial learning rate (adaptive process)
simopts.net.sigma       = simopts.net.size/2+1;     % initial neighborhood size (adaptive process)
simopts.net.maxepochs   = 100;                      % number of epochs to train
simopts.net.gamma       = 0.1;                      % cross-modal activation impact on local som learning
simopts.net.xi          = 0.1;                      % inhibitory component in sensory projections weight update
simopts.net.kappa       = 0.1;                      % learning rate (gain factor) in Hebbian weight update
simopts.net.lambda      = simopts.net.maxepochs/log(simopts.net.sigma); % temporal coef

%% RUN THE CORRELATION LEARNING NETWORK (MODES: RUN / ANALYZE)
% check mode 
switch(simopts.mode)
    case 'run'
        % prepare input data for the network
        netin = cln_sensory_dataset_setup(simopts);
        % create the SOM networks
        som1 = cln_create_som(simopts, netin.raw1, 'som1');
        som2 = cln_create_som(simopts, netin.raw2, 'som2');
        % run the network
        runtime_data = cln_iterate_network(simopts, netin, som1, som2);
    case 'analyze'
        % parametrize the simulation according the data file
        simopts.data.infile = sprintf('%d_epochs_%d_neurons_%s', simopts.net.maxepochs, simopts.net.size, simopts.data.corrtype);
        % prepare input data for visualization
        visin = cln_runtime_dataset_setup(simopts);
        % visualize network dynamics
        vishdl = cln_visualize(visin);
end
% end simulation