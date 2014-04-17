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

%% RUNTIME FLAGS

% visualization of the input data
input_vis = 0;

%% INPUT DATA VISUALIZATION

if(input_vis==1)
    % visualize input data
    in_vis = figure(1);
    set(gcf, 'color', 'white');
    subplot(2, 1, 1);plot(p1, '.b'); box off; grid off;
    title('Input var P1 (rate of change)');
    subplot(2, 1, 2);plot(p2, '.g'); box off; grid off;
    title('Input var P2 (accumulated values shifted in freq)');
    ylabel('Samples');
end

%% NETWORK STRUCTURE

% net parameters for structure and learning
NET_SIZE_LONG = 10;  % network lattice size long
NET_SIZE_LAT  = 10;  % network lattice size wide
% for rectangular lattice
NET_SIZE      = 10;
ALPHA0        = 0.1; % learning rate initialization
SIGMA0        = max(NET_SIZE_LONG, NET_SIZE_LAT)/2; % intial radius size
NET_ITER      = 1;  % inti counter for input vector entries
IN_SIZE       = 10; % input vector size = samples to bind in a input vector
MAX_EPOCHS    = 3; % epochs to run
LAMBDA        = 1000/log(SIGMA0); % time constant for radius adaptation
net_epochs    = 1;  % init counter for epochs
% extract the bound of the input intervals in the two input vars to
% initialize the weights in the bounds for faster convergence
% MIN_P1 = min(p1); MAX_P1 = max(p1);
% MIN_P2 = min(p2); MAX_P2 = max(p2);

MIN_P1 = 0; MAX_P1 = 1;
MIN_P2 = 0; MAX_P2 = 1;

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
bmu1=     struct('xpos', 0,...
    'ypos', 0,...
    'W'   , zeros(1, IN_SIZE),...          % input weights
    'H'   , zeros(NET_SIZE, NET_SIZE),...  % Hebbian weights for cross-SOM interaction
    'ad'  , 0.0, ... % direct activation elicited by input vector
    'ai'  , 0.0, ... % indirect activation elicited by cross-SOM interaction
    'at'  , 0.0 ...  % total joint activation (direct + indirect)
    );

for in_idx = 1:IN_SIZE
    bmu1.W(in_idx) = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
end
for kidx = 1:NET_SIZE
    for tidx = 1:NET_SIZE
        bmu1.H(kidx, tidx)  = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
    end
end
bmu1.ad = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
bmu1.ai = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
bmu1.at = MIN_P1 + (MAX_P1 - MIN_P1)*rand;

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
bmu2=     struct('xpos', 0,...
    'ypos', 0,...
    'W'   , zeros(1, IN_SIZE),...          % input weights
    'H'   , zeros(NET_SIZE, NET_SIZE),...  % Hebbian weights for cross-SOM interaction
    'ad'  , 0.0, ... % direct activation elicited by input vector
    'ai'  , 0.0, ... % indirect activation elicited by cross-SOM interaction
    'at'  , 0.0 ...  % total joint activation (direct + indirect)
    );

for in_idx = 1:IN_SIZE
    bmu2.W(in_idx) = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
end
for kidx = 1:NET_SIZE
    for tidx = 1:NET_SIZE
        bmu2.H(kidx, tidx)  = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
    end
end
bmu2.ad = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
bmu2.ai = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
bmu2.at = MIN_P2 + (MAX_P2 - MIN_P2)*rand;

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
qe1 = zeros(NET_SIZE, NET_SIZE);
qe2 = zeros(NET_SIZE, NET_SIZE);
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
sum_norm_W1 = 0.0; sum_norm_W2 = 0.0;
sum_norm_H1 = 0.0; sum_norm_H2 = 0.0;

% cross modal influence factor
GAMA = 0.1;
% inhibitory component weight in weight update
XI = 0.3;
% gain factor in Hebbian weight update
KAPPA = 0.4;

% training phase
while(1)
    if(net_epochs <= MAX_EPOCHS)
        % index of each entry in the sampled input vectors
        % we have length(in)/IN_SIZE vectors of size IN_SIZE
        for data_idx = 1:training_set_size
            % max quantization error init
            qe_max1 = 999999999;
            qe_max2 = 999999999;
            % search for the BMU in each SOM after applying an input vector
            % to the networks in a pairwise manner
            for idx = 1:NET_SIZE
                for jdx = 1:NET_SIZE
                    % compute the quantization error between the current
                    % input and the neurons in each SOM
                    % first to check is SOM1
                    qe1(idx, jdx) = norm(training_set_p1(data_idx, :) - som1(idx, jdx).W);
                    % check if current neuron in bmu
                    if(qe1(idx, jdx)<qe_max1)
                        bmu1.xpos = idx;
                        bmu1.ypos = jdx;
                        qe_max1 = qe1(idx, jdx);
                        bmu1_dist = qe1(idx, jdx);
                    end
                    % now check SOM2
                    qe2(idx, jdx) = norm(training_set_p2(data_idx, :) - som2(idx, jdx).W);
                    % check if current neuron in bmu
                    if(qe2(idx, jdx)<qe_max2)
                        bmu2.xpos = idx;
                        bmu2.ypos = jdx;
                        qe_max2 = qe2(idx, jdx);
                        bmu2_dist = qe2(idx, jdx);
                    end
                end
            end % end for BMU search loop
            
            % compute the activations of all nodes in the BMU neighborhood
            for idx = 1:NET_SIZE
                for jdx = 1:NET_SIZE
                    %-------------------------------------------------------------------------------
                    % use the same leraning parameters for both SOM
                    % compute the learning rate
                    ALPHA(net_epochs) = ALPHA0*exp(-net_epochs/TAU);
                    % compute the neighborhood radius size
                    SIGMA(net_epochs) = SIGMA0*exp(-net_epochs/LAMBDA);
                    %-------------------------------------------------------------------------------
                    % fist SOM activations
                    % compute the direct activation - neighborhood kernel
                    som1(idx, jdx).ad = ALPHA(net_epochs)*...
                        1/(sqrt(2*pi)*SIGMA(net_epochs))*...
                        exp(-(norm([idx - bmu1.xpos, jdx - bmu1.ypos]))^2/(2*(SIGMA(net_epochs)^2)))*(1 - qe1(idx, jdx));
                    
                    % compute the indirect activation (from all other units in SOM2)
                    % first compute the total activation from the other SOM
                    % via the Hebbian links
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
                    
                    % get the activation of the BMU from direct projection
                    % from the input space
                    bmu1_ad = som1(bmu1.xpos, bmu1.ypos).ad;
                    bmu1_ai = som1(bmu1.xpos, bmu1.ypos).ai;
                    som1(idx, jdx).at = (1 - GAMA)*1/(sqrt(2*pi)*SIGMA(net_epochs))*...
                        exp(-(norm([idx - bmu1.xpos, jdx - bmu1.ypos]))^2/(2*(SIGMA(net_epochs)^2))) + ...
                        GAMA* 1/(sqrt(2*pi)*SIGMA(net_epochs))*...
                        exp(-(norm([idx - bmu1.xpos, jdx - bmu1.ypos]))^2/(2*(SIGMA(net_epochs)^2)));
                    
                    % update weights for the current neuron in the BMU
                    % neighborhood if in the radius
                    if((norm([idx - bmu1.xpos, jdx - bmu1.ypos])) < SIGMA(net_epochs)^2)
                        
                        % normalize weights from input space
                        % compute the sum squared weight update for normalization
                        for w_idx = 1:IN_SIZE
                            for norm_idx = 1:NET_SIZE
                                for norm_jdx = 1:NET_SIZE
                                    sum_norm_W1 = sum_norm_W1 + (som1(norm_idx, norm_jdx).W(w_idx) + ALPHA(net_epochs)*som1(norm_idx, norm_jdx).at*qe1(norm_idx, norm_jdx)-...
                                        XI*(som1(norm_idx, norm_jdx).ad - som1(norm_idx, norm_jdx).at)*(training_set_p1(data_idx, w_idx) - som1(norm_idx, norm_jdx).W(w_idx)))^2;
                                end
                            end
                        end
                        
                        % input weights update combining an excitatory and
                        % inhibitory component such that a unit is brought
                        % closer to the input if is activated by BOTH input and
                        % cross modal input
                        for w_idx = 1:IN_SIZE
                            som1(idx, jdx).W(w_idx) = (som1(idx, jdx).W(w_idx) + ALPHA(net_epochs)*som1(idx, jdx).at*qe1(idx, jdx)-...
                                XI*(som1(idx, jdx).ad - som1(idx, jdx).at)*(training_set_p1(data_idx, w_idx) - som1(idx, jdx).W(w_idx)))/...
                                sqrt(sum_norm_W1);
                        end
                    end % end if in BMU neighborhood
                    
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                    for isom2 = 1:NET_SIZE
                        for jsom2 = 1:NET_SIZE
                            
                            % normalize cross-modal Hebbian weights
                            for norm_idx = 1:NET_SIZE
                                for norm_jdx = 1:NET_SIZE
                                    sum_norm_H1 = sum_norm_H1 + (som1(idx, jdx).H(isom2, jsom2)+KAPPA*(som1(idx, jdx).at*som2(idx, jdx).at))^2;
                                end
                            end
                            
                            % compute new weight
                            som1(idx, jdx).H(isom2, jsom2)= (som1(idx, jdx).H(isom2, jsom2)+KAPPA*(som1(idx, jdx).at*som2(idx, jdx).at))/...
                                sqrt(sum_norm_H1);
                        end
                    end
                    
                    %-------------------------------------------------------------------------------
                    % second SOM activations
                    
                    % compute the direct activation - neighborhood kernel
                    som2(idx, jdx).ad = ALPHA(net_epochs)*...
                        1/(sqrt(2*pi)*SIGMA(net_epochs))*...
                        exp(-(norm([idx - bmu2.xpos, jdx - bmu2.ypos]))^2/(2*(SIGMA(net_epochs)^2)))*(1-qe2(idx, jdx));
                    
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
                    
                    % get the activation of the BMU from direct projection
                    % from the input space
                    bmu2_ad = som2(bmu2.xpos, bmu2.ypos).ad;
                    bmu2_ai = som2(bmu2.xpos, bmu2.ypos).ai;
                    som2(idx, jdx).at = (1 - GAMA)*1/(sqrt(2*pi)*SIGMA(net_epochs))*...
                        exp(-(norm([idx - bmu2.xpos, jdx - bmu2.ypos]))^2/(2*(SIGMA(net_epochs)^2))) + ...
                        GAMA* 1/(sqrt(2*pi)*SIGMA(net_epochs))*...
                        exp(-(norm([idx - bmu2.xpos, jdx - bmu2.ypos]))^2/(2*(SIGMA(net_epochs)^2)));
                    
                    % update weights for the current neuron in the BMU
                    % neighborhood if in the radius
                    if((norm([idx - bmu2.xpos, jdx - bmu2.ypos])) < SIGMA(net_epochs)^2)
                        
                        % normalize weights from input space
                        % compute the sum squared weight update for normalization
                        for w_idx = 1:IN_SIZE
                            for norm_idx = 1:NET_SIZE
                                for norm_jdx = 1:NET_SIZE
                                    sum_norm_W2 = sum_norm_W2 + (som2(norm_idx, norm_jdx).W(w_idx) + ALPHA(net_epochs)*som2(norm_idx, norm_jdx).at*qe2(norm_idx, norm_jdx)-...
                                        XI*(som2(norm_idx, norm_jdx).ad - som2(norm_idx, norm_jdx).at)*(training_set_p2(data_idx, w_idx) - som2(norm_idx, norm_jdx).W(w_idx)))^2;
                                end
                            end
                        end
                        
                        % input weights update combining an excitatory and
                        % inhibitory component such that a unit is brought
                        % closer to the input if is activated by BOTH input and
                        % cross modal input
                        for w_idx = 1:IN_SIZE
                            som2(idx, jdx).W(w_idx) = (som2(idx, jdx).W(w_idx) + ALPHA(net_epochs)*som2(idx, jdx).at*qe2(idx, jdx)-...
                                XI*(som2(idx, jdx).ad - som2(idx, jdx).at)*(training_set_p2(data_idx, w_idx) - som2(idx, jdx).W(w_idx)))/...
                                sqrt(sum_norm_W2);
                            
                        end
                    end % end if in BMU neighborhood
                    
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                    for isom1 = 1:NET_SIZE
                        for jsom1 = 1:NET_SIZE
                            
                            % normalize cross-modal Hebbian weights
                            for norm_idx = 1:NET_SIZE
                                for norm_jdx = 1:NET_SIZE
                                    sum_norm_H2 = sum_norm_H2 + (som2(idx, jdx).H(isom1, jsom1)+KAPPA*(som2(idx, jdx).at*som1(idx, jdx).at))^2;
                                end
                            end
                            
                            % compute new weight
                            som2(idx, jdx).H(isom1, jsom1)= (som2(idx, jdx).H(isom1, jsom1)+KAPPA*(som2(idx, jdx).at*som1(idx, jdx).at))/...
                                sqrt(sum_norm_H2);
                        end
                    end
                    
                    %-------------------------------------------------------------------------------
                end
            end
            
        end % for each entry in the training set
        % increment epochs counter
        net_epochs = net_epochs + 1;
    else
        disp 'Finalized learning phase.';
        break;
    end
end
