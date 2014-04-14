% simple temporal correlation learning mechanism using SOM
% learn the temporal correlations between 2 variables 

%% INPUT DATA 
% create the 2 data sets 
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

% visualize input data 
in_vis = figure(1);
set(gcf, 'color', 'white'); 
subplot(2, 1, 1);plot(p1, '.b'); box off; grid off;
title('Input var P1 (rate of change)'); 
subplot(2, 1, 2);plot(p2, '.g'); box off; grid off;
title('Input var P2 (accumulated values shifted in freq)'); 
ylabel('Samples');

%% NETWORK STRUCTURE
% net parameters for structure and learning
NET_SIZE_LONG = 10;  % network lattice size long
NET_SIZE_LAT  = 10;  % network lattice size wide
% for rectangular lattice
NET_SIZE      = 10;
ALPHA0        = 0.2; % learning rate initialization
SIGMA0        = max(NET_SIZE_LONG, NET_SIZE_LAT)/2; % intial radius size
NET_ITER      = 1;  % inti counter for input vector entries
IN_SIZE       = 10; % input vector size = samples to bind in a input vector
MAX_EPOCHS    = 10; % epochs to run
LAMBDA        = MAX_EPOCHS/log(SIGMA0); % time constant for radius adaptation
net_epochs    = 1;  % init counter for epochs
% extract the bound of the input intervals in the two input vars to
% initialize the weights in the bounds for faster convergence
MIN_P1 = min(p1); MAX_P1 = max(p1);
MIN_P2 = min(p2); MAX_P2 = max(p2);

%% INITIALIZE THE SOM FOR THE FIRST INPUT

% create struct for first SOM
som1(1:NET_SIZE, 1:NET_SIZE) = struct('xpos', 0,...
                                             'ypos', 0,...
                                             'W'   , zeros(1, IN_SIZE),...          % input weights
                                             'H'   , zeros(NET_SIZE, NET_SIZE),...  % Hebbian weights for cross-SOM interaction
                                             'ad'  , zeros(NET_SIZE, NET_SIZE), ... % direct activation elicited by input vector
                                             'ai'  , zeros(NET_SIZE, NET_SIZE), ... % indirect activation elicited by cross-SOM interaction
                                             'at'  , zeros(NET_SIZE, NET_SIZE) ...  % total joint activation (direct + indirect)
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
                som1(idx, jdx).ad(kidx, tidx) = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
                som1(idx, jdx).ai(kidx, tidx) = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
                som1(idx, jdx).at(kidx, tidx) = MIN_P1 + (MAX_P1 - MIN_P1)*rand;
            end
        end
    end
end

%% INITIALIZE THE SOM FOR SECOND INPUT

% create struct for second SOM
som2(1:NET_SIZE, 1:NET_SIZE) = struct('xpos', 0,...
                                             'ypos', 0,...
                                             'W'   , zeros(1, IN_SIZE),...          % input weights
                                             'H'   , zeros(NET_SIZE, NET_SIZE),...  % Hebbian weights for cross-SOM interaction
                                             'ad'  , zeros(NET_SIZE, NET_SIZE), ... % direct activation elicited by input vector
                                             'ai'  , zeros(NET_SIZE, NET_SIZE), ... % indirect activation elicited by cross-SOM interaction
                                             'at'  , zeros(NET_SIZE, NET_SIZE) ...  % total joint activation (direct + indirect)
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
                som2(idx, jdx).ad(kidx, tidx) = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
                som2(idx, jdx).ai(kidx, tidx) = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
                som2(idx, jdx).at(kidx, tidx) = MIN_P2 + (MAX_P2 - MIN_P2)*rand;
            end
        end
    end
end

%% NETWORK DYNAMICS
% in the same loop train SOM1 and SOM2 and cross-SOM interaction
% the units which will be attracted towards the inputs will be the ones
% which have activation from both input vector as well as cross-modal
% activation

% training phase
while(1)
    if(net_epochs <= MAX_EPOCHS)
        
        
    else
        disp 'Finalized learning phase.';
        break;
    end
end

