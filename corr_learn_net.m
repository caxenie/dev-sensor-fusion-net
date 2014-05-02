%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% Main simulation script
%% SETUP ENVIRONMENT

clear all; close all; clc
%% LOAD DATA AND SETUP RUNTIME

% simulation options parametrization
simopts.mode = 'run'; % mode given the function of the script, i.e. run, analyze
simopts.data.infile = 'robot_data_jras_paper';
simopts.data.scaling = -0.0572957795130823; % data dependent scaling
simopts.data.freqsamp = 25; % Hz
simopts.data.trainvtype = 'interval'; % train vector type, i.e. fixed interval / sliding window
simopts.data.trainvsize = 100; % size (in samples) of the input vector
simopts.data.corrtype = 'algebraic'; % input data correlation type, i.e. algebraic, temporal, nonlinear
% parametrize the network
simopts.net.size = 10; % 10 x 10 square lattice SOM nets
simopts.net.alpha = 0.1; % initial learning rate (adaptive process)
simopts.net.sigma = simopts.net.size/2+1; % initial neighborhood size (adaptive process)
simopts.net.maxepochs = 30; % number of epochs to train
simopts.net.gamma = 0.35; % cross-modal activation impact on local som learning
simopts.net.xi = 0.27; % inhibitory component in sensory projections weight update
simopts.net.kappa = 0.23; % learning rate (gain factor) in Hebbian weight update
simopts.net.lambda = simopts.net.maxepochs/log(simopts.net.sigma); % temporal coef
% prepare input data for the network
netin = cln_sensory_dataset_setup(simopts);
% create the SOM networks
som1 = cln_create_som(simopts, netin.raw1);
som2 = cln_create_som(simopts, netin.raw2);

%% RUN THE CORRELATION LEARNING NETWORK (MODES: RUN / ANALYZE)
% check mode
switch(simopts.mode)
    case 'run'
        runtime_data = cln_iterate_network(simopts, netin, som1, som2);
    case 'analyze'
        % parametrize the simulation according the data
        simopts.data.infile = '10_epochs_algebraic';
        % prepare input data for visualization
        visin = cln_runtime_dataset_setup(simopts);
        % VISUALIZE THE NETWORK DYNAMICS
        vishdl = cln_visualize(visin);
end
% end simulation