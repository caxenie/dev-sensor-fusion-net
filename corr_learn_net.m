%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% Main simulation script
%% SETUP ENVIRONMENT
clear all; close all; clc; pause(2);
%% LOAD DATA AND SETUP RUNTIME
% -------------- simulation options parametrization ---------------------
simopts.mode            = 'run';                      % mode given the function of the script, i.e. run, analyze
simopts.debug.verbose   = 0;                          % flag to activate / inactivate debug verbose
simopts.debug.visual    = 0;                          % flag to activate / inactivate debug visualization
% ---------- data generation and preprocessing parametrization ----------
simopts.data.source     = 'generated';                % data source: generated or sensors (data from robot)
simopts.data.trainvtype = 'full';                  % train vector type, i.e. fixed interval / sliding window / full dataset
simopts.data.slidesize  = 1;                          % sliding window size (only for sliding) slidesize < trainvsize
simopts.data.numsamples = 1000;                        % total number of samples to generate (only for generated data)
simopts.data.trainvsize = 1000;                         % size (in samples) of the input vector (only for sliding and interval)
simopts.data.ntrainv    = 100;                        % number of trin vectors for training the network (only for full dataset)
simopts.data.corrtype   = 'algebraic';                % input data correlation type, i.e. algebraic, temporal, nonlinear, delay (sine waves only for generated), amplitude (sine waves only for generated)
% ---------------------- parametrize the network ------------------------
simopts.net.size        = 5;                          % size x size square lattice SOM nets
simopts.net.params      = 'adaptive';                 % adaptive processes parameters, i.e. fixed/adaptive
simopts.net.alpha       = 0.1;                        % initial learning rate (adaptive process)
simopts.net.sigma       = simopts.net.size/2+1;       % initial neighborhood size (adaptive process)
simopts.net.maxepochs   = 500;                        % number of epochs to train
simopts.net.gamma       = 0.1;                        % cross-modal activation impact on local som learning
simopts.net.xi          = 0.01;                       % inhibitory component in sensory projections weight update
simopts.net.kappa       = 0.2;                        % learning rate (gain factor) in Hebbian weight update
simopts.net.lambda      = simopts.net.maxepochs/...
                          log(simopts.net.sigma);     % temporal coef
simopts.net.xmodlearn   = 'hebb';                     % cross modal learning mechanism, i.e. hebb or covariance (pseudo-Hebbian)
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
        simopts.data.infile = sprintf('%d_epochs_%d_neurons_%s_source_data_%s_correlation_%d_trainvsize_%d_trainvnum_%s_params',...
            simopts.net.maxepochs, simopts.net.size, simopts.data.source, simopts.data.corrtype, simopts.data.trainvsize,...
            simopts.data.ntrainv, simopts.net.params);
        % prepare input data for visualization
        visin = cln_runtime_dataset_setup(simopts);
        if(isempty(fieldnames(visin))==1) 
            return; % no net runtime data file present
        end
        % visualize network dynamics
        cln_visualize(visin);
        fprintf(1, 'Finalized visualization process\n');
end
% end simulation 