% simple temporal correlation learning mechanism using SOM
% learn the temporal correlations between 2 variables

%% INPUT DATA
% create the 2 data sets
clear all; close all; clc;
test_data = load('robot_data_jras_paper');

% first variable (derivative - rate of change @ 20Hz, e.g. gyro)
scaling_factor = 1000;
% scale the data
p1 = (test_data(:,7)*(180/pi))/scaling_factor;
p1 = -p1;

% test flag to implement different raltionships between the 2 encoded
% variables in the network

% types of relations
temporal = 1;
algebraic = 2;
trigo = 3;

c_type = [temporal, algebraic, trigo];
c_type_idx = length(c_type);

while(c_type_idx>=1)
% set type
correlation_type = c_type(c_type_idx);

switch correlation_type
    case temporal
        % second variable (integral - accumulation @ 25Hz -> shift in freq)
        sample_freq = 25; % Hz
        sample_time = 1/sample_freq;
        p2 = zeros(1, length(test_data));
        p2(1) = p1(1);
        % compute the absolute value (Euler)
        for i=2:length(test_data)
            p2(i) = p2(i-1) + ...
                (sample_time*(p1(i)));
        end
        % measure the cross-correlation between the vars
        temporal_xcorr = xcorr(p1, p2);
        p2_temporal = p2;
    case algebraic
        % second variable might be linked to the first one using an algebraic
        % relationship - sample
        p2 = 3*p1 - 5;
        % measure the cross-correlation between the vars
        algebraic_xcorr = xcorr(p1, p2);
        p2_algebraic = p2;
    case trigo
        % second varible is linked to the first using a trigonometric
        % relationship - sample
        p2 = 4.5*sin(p1/5) + 6.5;
        % measure the cross-correlation between the vars
        trigo_xcorr = xcorr(p1, p2);
        p2_trigo = p2;
end % type switch
% switch type
c_type_idx = c_type_idx - 1;
end

%% RUNTIME FLAGS

% visualization of the input data
verbose = 0;
% visualization of the correlation analysis
xcorr_verbose = 1;

%% INPUT DATA VISUALIZATION

if(verbose==1)
    % visualize input data
    in_vis = figure(1);
    set(gcf, 'color', 'white');
    subplot(2, 1, 1);plot(p1, '.b'); box off; grid off;
    title('Input var P1');
    subplot(2, 1, 2);plot(p2, '.g'); box off; grid off;
    title('Input var P2');
    ylabel('Samples');
end

if(xcorr_verbose==1)
    % visualize cross-correlation
    xcorr_vis = figure(2);
    set(gcf, 'color', 'white');
    subplot(6, 2, 1);plot(p1, '.b'); box off; grid off;
    title('P1 var');
    subplot(6, 2, 3);plot(p2_temporal, '.b'); box off; grid off;
    title('P2 var');
    subplot(6, 2, [2,4]);plot(temporal_xcorr, '.b'); box off; grid off;
    title('Temporal xcorr analysis');
    
    subplot(6, 2, 5);plot(p1, '.g'); box off; grid off;
    title('P1 var');
    subplot(6, 2, 7);plot(p2_algebraic, '.g'); box off; grid off;
    title('P2 var');
    subplot(6, 2, [6,8]);plot(algebraic_xcorr, '.g'); box off; grid off;
    title('Algebraic xcorr analysis');
    
    subplot(6, 2, 9);plot(p1, '.r'); box off; grid off;
    title('P1 var');
    subplot(6, 2, 11);plot(p2_trigo, '.r'); box off; grid off;
    title('P2 var');
    subplot(6, 2, [10,12]);plot(trigo_xcorr, '.r'); box off; grid off;
    title('Trigonometric xcorr analysis');
    ylabel('Samples');
end

return
%% NETWORK STRUCTURE

% for rectangular lattice
NET_SIZE      = 5;
% net parameters for structure and learning
NET_SIZE_LONG = NET_SIZE;  % network lattice size long
NET_SIZE_LAT  = NET_SIZE;  % network lattice size wide
ALPHA0        = 0.1; % learning rate initialization
SIGMA0        = max(NET_SIZE_LONG, NET_SIZE_LAT)/2 + 1; % intial radius size
IN_SIZE       = 10; % input vector size = samples to bind in the input vector
MAX_EPOCHS    = 500*NET_SIZE; % epochs to run
LAMBDA        = MAX_EPOCHS/log(SIGMA0); % time constant for radius adaptation

% iterator for relaxation
net_epochs    = 1;  % init counter for epochs

% extract the bound of the input intervals in the two input vars to
% initialize the weights in the bounds for faster convergence

% for temporal computation it's good to speed up computation by
% initializing the weights in the min - max bounds of the data from each
% source
if correlation_type == temporal
    MIN_P1 = min(p1); MAX_P1 = max(p1);
    MIN_P2 = min(p2); MAX_P2 = max(p2);
else
    % else a [0,1] initialization is fine
    MIN_P1 = 0; MAX_P1 = 1;
    MIN_P2 = 0; MAX_P2 = 1;
end

%% INITIALIZE THE SOM FOR THE FIRST INPUT

% create struct for first SOM
som1(1:NET_SIZE, 1:NET_SIZE) = struct('xpos', 0,...
    'ypos', 0,...
    'W'   , zeros(1, IN_SIZE),...          % input weights
    'H'   , zeros(NET_SIZE,NET_SIZE),...  % Hebbian weights for cross-SOM interaction
    'ad'  , 0.0, ... % direct activation elicited by input vector
    'ai'  , 0.0, ... % indirect activation elicited by cross-SOM interaction
    'at'  , 0.0 ...  % total joint activation (direct + indirect)
    );
% initialize first SOM
for idx = 1:NET_SIZE
    for jdx = 1:NET_SIZE
        som1(idx, jdx).xpos = idx;
        som1(idx, jdx).ypos = jdx;
        for in_idx = 1:IN_SIZE
            som1(idx, jdx).W(in_idx) = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
        end
        for kidx = 1:NET_SIZE
            for tidx = 1:NET_SIZE
                som1(idx, jdx).H(kidx, tidx)  = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
            end
        end
        som1(idx, jdx).ad = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
        som1(idx, jdx).ai = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
        som1(idx, jdx).at = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
    end
end

% intialize the SOM1 BMU
bmu1_dir=     struct('xpos', 0,...
    'ypos', 0,...
    'W'   , zeros(1, IN_SIZE),...          % input weights
    'H'   , zeros(NET_SIZE, NET_SIZE),...  % Hebbian weights for cross-SOM interaction
    'ad'  , 0.0, ... % direct activation elicited by input vector
    'ai'  , 0.0, ... % indirect activation elicited by cross-SOM interaction
    'at'  , 0.0 ...  % total joint activation (direct + indirect)
    );

for in_idx = 1:IN_SIZE
    bmu1_dir.W(in_idx) = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
end
for kidx = 1:NET_SIZE
    for tidx = 1:NET_SIZE
        bmu1_dir.H(kidx, tidx)  = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
    end
end
bmu1_dir.ad = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
bmu1_dir.ai = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
bmu1_dir.at = MIN_P1 + (MAX_P1 - MIN_P1)*rand;

%% INITIALIZE THE SOM FOR SECOND INPUT

% create struct for second SOM
som2(1:NET_SIZE, 1:NET_SIZE) = struct('xpos', 0,...
    'ypos', 0,...
    'W'   , zeros(1, IN_SIZE),...          % input weights
    'H'   , zeros(NET_SIZE,NET_SIZE),...  % Hebbian weights for cross-SOM interaction
    'ad'  , 0.0, ... % direct activation elicited by input vector
    'ai'  , 0.0, ... % indirect activation elicited by cross-SOM interaction
    'at'  , 0.0 ...  % total joint activation (direct + indirect)
    );
% initialize second SOM
for idx = 1:NET_SIZE
    for jdx = 1:NET_SIZE
        som2(idx, jdx).xpos = idx;
        som2(idx, jdx).ypos = jdx;
        for in_idx = 1:IN_SIZE
            som2(idx, jdx).W(in_idx) = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
        end
        for kidx = 1:NET_SIZE
            for tidx = 1:NET_SIZE
                som2(idx, jdx).H(kidx, tidx)  = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
            end
        end
        som2(idx, jdx).ad = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
        som2(idx, jdx).ai = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
        som2(idx, jdx).at = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
    end
end

% intialize the SOM1 BMU
bmu2_dir=     struct('xpos', 0,...
    'ypos', 0,...
    'W'   , zeros(1, IN_SIZE),...          % input weights
    'H'   , zeros(NET_SIZE, NET_SIZE),...  % Hebbian weights for cross-SOM interaction
    'ad'  , 0.0, ... % direct activation elicited by input vector
    'ai'  , 0.0, ... % indirect activation elicited by cross-SOM interaction
    'at'  , 0.0 ...  % total joint activation (direct + indirect)
    );

for in_idx = 1:IN_SIZE
    bmu2_dir.W(in_idx) = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
end
for kidx = 1:NET_SIZE
    for tidx = 1:NET_SIZE
        bmu2_dir.H(kidx, tidx)  = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
    end
end
bmu2_dir.ad = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
bmu2_dir.ai = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
bmu2_dir.at = MIN_P2 + (MAX_P2 - MIN_P2)*rand;

%% NETWORK DYNAMICS

% split the input vectors in training vectors of size IN_SIZE
training_set_size = round(length(p1)/IN_SIZE);
training_set_p1 = zeros(round(length(p1)/IN_SIZE), IN_SIZE);
training_set_p2 = zeros(round(length(p2)/IN_SIZE), IN_SIZE);

training_set_p1(1, :) = p1(1:IN_SIZE);
training_set_p2(1, :) = p2(1:IN_SIZE);

% fill the training datasets
for idx = 2:training_set_size
    for jdx = 1:IN_SIZE
        training_set_p1(idx, jdx) = p1(((idx-1)*IN_SIZE + jdx));
        training_set_p2(idx, jdx) = p2(((idx-1)*IN_SIZE + jdx));
    end
end

% in the same loop train SOM1 and SOM2 and cross-SOM interaction
% the units which will be attracted towards the inputs will be the ones
% which have activation from boh input vector as well as cross-modal
% activation

% init quantization errors
qe1_dir = zeros(NET_SIZE, NET_SIZE);
qe2_dir = zeros(NET_SIZE, NET_SIZE);
qe1_ind = zeros(NET_SIZE, NET_SIZE);
qe2_ind = zeros(NET_SIZE, NET_SIZE);
bmu1_dist = 0.0;
bmu2_dist = 0.0;

% init learning rate and radius
ALPHA = zeros(1, MAX_EPOCHS); TAU = 1000; % for learning rate adaptation
SIGMA = zeros(1, MAX_EPOCHS);

% cross modal influences init
cross_mod1 = 0.0;
cross_mod2 = 0.0;

% normalization factors
% for input weights
sum_norm_W1 = zeros(1, IN_SIZE); sum_norm_W2 = zeros(1, IN_SIZE);
sum_norm_H1 = zeros(1, IN_SIZE); sum_norm_H2 = zeros(1, IN_SIZE);

% cross modal influence factor
GAMA = 0.35;
% inhibitory component weight in weight update
XI = 0.27;
% gain factor in Hebbian weight update
KAPPA = 0.23;

% training phase
while(1)
    if(net_epochs <= MAX_EPOCHS)
        % start timer
        start_epoch = tic;
        % index of each entry in the sampled input vectors
        % we have length(in)/IN_SIZE vectors of size IN_SIZE
        for data_idx = 1:training_set_size
            % start timer for input vector entries presented to the net
            start_input_entry  = tic;
            % max quantization error init
            qe_max1_dir = Inf; qe_max2_dir = Inf;
            qe_max1_ind = Inf; qe_max2_ind = Inf;
            
            % search for the BMU in each SOM after applying an input vector
            % this is the BMU from input propagation
            % to the networks in a pairwise manner
            for idx = 1:NET_SIZE
                for jdx = 1:NET_SIZE
                    
                    % compute the quantization error between the current
                    % input and the neurons in each SOM
                    % first to check is SOM1
                    qe1_dir(idx, jdx) = norm(training_set_p1(data_idx, :) - som1(idx, jdx).W);
                    % check if current neuron is winner
                    if(qe1_dir(idx, jdx)<qe_max1_dir)
                        bmu1_dir.xpos = idx;
                        bmu1_dir.ypos = jdx;
                        qe_max1_dir = qe1_dir(idx, jdx);
                        bmu1_dist = qe1_dir(idx, jdx);
                    end
                    % now check SOM2
                    qe2_dir(idx, jdx) = norm(training_set_p2(data_idx, :) - som2(idx, jdx).W);
                    % check if current neuron is winner
                    if(qe2_dir(idx, jdx)<qe_max2_dir)
                        bmu2_dir.xpos = idx;
                        bmu2_dir.ypos = jdx;
                        qe_max2_dir = qe2_dir(idx, jdx);
                        bmu2_dist = qe2_dir(idx, jdx);
                    end
                end
                
                % we also need to compute the BMU from the cross-modal
                % interaction between the maps
                
                % compute the cross-interaction BMU for the first SOM
                for som2_idx = 1:NET_SIZE
                    for som2_jdx = 1:NET_SIZE
                        qe1_ind(idx, jdx) = norm(som2(som2_idx, som2_jdx).H - som1(idx, jdx).H);
                        % check if current neuron is winner
                        if(qe1_ind(idx, jdx)<qe_max1_ind)
                            bmu1_ind.xpos = idx;
                            bmu1_ind.ypos = jdx;
                            qe_max1_ind = qe1_ind(idx, jdx);
                        end
                    end
                end
                
                % compute the cross-interaction BMU for the seond SOM
                for som1_idx = 1:NET_SIZE
                    for som1_jdx = 1:NET_SIZE
                        qe2_ind(idx, jdx) = norm(som1(som1_idx, som1_jdx).H - som2(idx, jdx).H);
                        % check if current neuron is winner
                        if(qe2_ind(idx, jdx)<qe_max2_ind)
                            bmu2_ind.xpos = idx;
                            bmu2_ind.ypos = jdx;
                            qe_max2_ind = qe2_ind(idx, jdx);
                        end
                    end
                end
                
            end % end for BMU search loop
            
            % compute the activations of all nodes in the BMU neighborhood
            for idx = 1:NET_SIZE
                for jdx = 1:NET_SIZE
                    %-------------------------------------------------------------------------------
                    % use the same leraning parameters for both SOM
                    % compute the learning rate @ current epoch
                    
                    % exponential learning rate adaptation
                    % ALPHA(net_epochs) = ALPHA0*exp(-net_epochs/TAU);
                    % semi-empirical learning rate adaptation
                    A = MAX_EPOCHS/100.0; B = A;
                    ALPHA(net_epochs) = A/(net_epochs + B);
                    
                    % compute the neighborhood radius size @ current epoch
                    SIGMA(net_epochs) = SIGMA0*exp(-net_epochs/LAMBDA);
                    %-------------------------------------------------------------------------------
                    % fist SOM activations
                    % compute the direct activation - neighborhood kernel
                    som1(idx, jdx).ad = ALPHA(net_epochs)*...
                        exp(-(norm([bmu1_dir.xpos - idx, bmu1_dir.ypos - jdx]))^2/(2*(SIGMA(net_epochs)^2)))*(1 - qe1_dir(idx, jdx));
                    
                    % compute the indirect activation (from all other units in SOM2)
                    % first compute the total activation from the other SOM
                    % projected via the Hebbian links
                    for isom2 = 1:NET_SIZE
                        for jsom2 = 1:NET_SIZE
                            % sum of all products between direct activation
                            % and cross-som Hebbian weights
                            cross_mod2 = cross_mod2 + ...
                                som2(isom2, jsom2).ad*som1(idx, jdx).H(isom2, jsom2);
                        end
                    end
                    som1(idx, jdx).ai = cross_mod2;
                    
                    % compute the joint activation from both input space
                    % and cross-modal Hebbian linkage
                    
                    som1(idx, jdx).at = (1 - GAMA)*exp(-(norm([bmu1_dir.xpos - idx, bmu1_dir.ypos - jdx]))^2/(2*(SIGMA(net_epochs)^2))) + ...
                        GAMA*exp(-(norm([bmu1_ind.xpos - idx,  bmu1_ind.ypos - jdx]))^2/(2*(SIGMA(net_epochs)^2)));
                    
                    % update weights for the current neuron in the BMU
                    
                    % normalize weights from input space
                    % compute the sum squared weight update for normalization
                    for w_idx = 1:IN_SIZE
                        for norm_idx = 1:NET_SIZE
                            for norm_jdx = 1:NET_SIZE
                                sum_norm_W1(w_idx) = sum_norm_W1(w_idx) + (som1(norm_idx, norm_jdx).W(w_idx) + ALPHA(net_epochs)*som1(norm_idx, norm_jdx).at*qe1_dir(norm_idx, norm_jdx)-...
                                    XI*(som1(norm_idx, norm_jdx).ad - som1(norm_idx, norm_jdx).at)*qe1_dir(norm_idx, norm_jdx))^2;
                            end
                        end
                    end
                    
                    % input weights update combining an excitatory and
                    % inhibitory component such that a unit is brought
                    % closer to the input if is activated by BOTH input and
                    % cross modal input
                    for w_idx = 1:IN_SIZE
                        som1(idx, jdx).W(w_idx) = (som1(idx, jdx).W(w_idx) + ALPHA(net_epochs)*som1(idx, jdx).at*qe1_dir(idx, jdx)-...
                            XI*(som1(idx, jdx).ad - som1(idx, jdx).at)*(training_set_p1(data_idx, w_idx) - som1(idx, jdx).W(w_idx)))/...
                            sqrt(sum_norm_W1(w_idx));
                    end
                    
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                    for isom2 = 1:NET_SIZE
                        for jsom2 = 1:NET_SIZE
                            
                            % compute new weight using Hebbian learning
                            % rule deltaH = K*preH*postH
                            som1(idx, jdx).H(isom2, jsom2)= som1(idx, jdx).H(isom2, jsom2)+ KAPPA*som1(idx, jdx).at*som2(idx, jdx).at;
                        end
                    end
                    
                    %-------------------------------------------------------------------------------
                    % compute second SOM activations
                    
                    % compute the direct activation - neighborhood kernel
                    som2(idx, jdx).ad = ALPHA(net_epochs)*...
                        exp(-(norm([bmu2_dir.xpos - idx, bmu2_dir.ypos - jdx]))^2/(2*(SIGMA(net_epochs)^2)))*(1-qe2_dir(idx, jdx));
                    
                    % compute the indirect activation (from all other units in SOM2)
                    % first compute the total activation from the other SOM
                    % via the Hebbian links
                    for isom1 = 1:NET_SIZE
                        for jsom1 = 1:NET_SIZE
                            % sum of all products between direct activation
                            % and cross-som Hebbian weights
                            cross_mod1 = cross_mod1 + ...
                                som1(isom1, jsom1).ad*som2(idx, jdx).H(isom1, jsom1);
                        end
                    end
                    som2(idx, jdx).ai = cross_mod1;
                    
                    % compute the joint activation from both input space
                    % and cross-modal Hebbian linkage
                    
                    som2(idx, jdx).at = (1 - GAMA)*exp(-(norm([idx - bmu2_dir.xpos, jdx - bmu2_dir.ypos]))^2/(2*(SIGMA(net_epochs)^2))) + ...
                        GAMA*exp(-(norm([idx - bmu2_ind.xpos, jdx - bmu2_ind.ypos]))^2/(2*(SIGMA(net_epochs)^2)));
                    
                    % update weights for the current neuron in the BMU
                    
                    % normalize weights from input space
                    % compute the sum squared weight update for normalization
                    for w_idx = 1:IN_SIZE
                        for norm_idx = 1:NET_SIZE
                            for norm_jdx = 1:NET_SIZE
                                sum_norm_W2(w_idx) = sum_norm_W2(w_idx) + (som2(norm_idx, norm_jdx).W(w_idx) + ALPHA(net_epochs)*som2(norm_idx, norm_jdx).at*qe2_dir(norm_idx, norm_jdx)-...
                                    XI*(som2(norm_idx, norm_jdx).ad - som2(norm_idx, norm_jdx).at)*qe2_dir(norm_idx, norm_jdx))^2;
                            end
                        end
                    end
                    
                    % input weights update combining an excitatory and
                    % inhibitory component such that a unit is brought
                    % closer to the input if is activated by BOTH input and
                    % cross modal input
                    for w_idx = 1:IN_SIZE
                        som2(idx, jdx).W(w_idx) = (som2(idx, jdx).W(w_idx) + ALPHA(net_epochs)*som2(idx, jdx).at*qe2_dir(idx, jdx)-...
                            XI*(som2(idx, jdx).ad - som2(idx, jdx).at)*(training_set_p2(data_idx, w_idx) - som2(idx, jdx).W(w_idx)))/...
                            sqrt(sum_norm_W2(w_idx));
                        
                    end
                    
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                    for isom1 = 1:NET_SIZE
                        for jsom1 = 1:NET_SIZE
                            
                            % update weight using Hebbian learning rule
                            som2(idx, jdx).H(isom1, jsom1)= som2(idx, jdx).H(isom1, jsom1) + KAPPA*som2(idx, jdx).at*som1(idx, jdx).at;
                        end
                    end
                    
                    %-------------------------------------------------------------------------------
                end
            end
            % elapsed time to propagate all values from input vector
            elapsed_time_input_vector = toc(start_input_entry);
            if(verbose==1)
                fprintf('Elapsed time for data entry %d is %d secs \n', data_idx, elapsed_time_input_vector);
            end
        end % for each entry in the training set
        
        % increment epochs counter
        net_epochs = net_epochs + 1;
        
        % elapsed time to propagate all values from input vector
        elapsed_time_epoch = toc(start_epoch);
        if(verbose==1)
            fprintf('--> Elapsed time for epoch %d is %d secs \n', net_epochs-1, elapsed_time_epoch);
        end
    else
        disp 'Finalized learning phase.';
        break;
    end
end
