% SOM based correlation learning for fusion network
% the SOM is used to learn the temporal correlation between the multiple
% sensory modalities presented to the network
% after learning the correlations the topology will be extracted and then
% the low-level dynamics will follow the generic dynamics

% the initial scenario is for heading estimation network development

% clear up environment 
clear all; clc;

% initialize the simulation 
% load the robot sensor dataset
robot_data = load('robot_data_jras_paper');
vision_data = load('tracker_data_jras_paper');

%% Datasets setup

% tolerance in timestamp comparison
tolerance = 1e-18;

% get the start and stop timestamp in both datasets
start_ts_vision = min(vision_data(:,1));
start_ts_robot   = min(robot_data(:,1));
stop_ts_vision  = max(vision_data(:,1));
stop_ts_robot    = max(robot_data(:,1));

% compute the global start and stop time for the data
global_start = max(start_ts_robot, start_ts_vision);
global_stop  = min(stop_ts_robot, stop_ts_vision);

% select and remove out-of-range entries for robot sensory dataset
pruned_entries_robot = [];
pruned_idx_robot = 1;
for i=1:length(robot_data)
    % start check
    if (robot_data(i,1) - global_start < tolerance)
        pruned_entries_robot(pruned_idx_robot) = i;
        pruned_idx_robot = pruned_idx_robot + 1;
    end
    % end check
    if (robot_data(i,1) - global_stop > tolerance)
        pruned_entries_robot(pruned_idx_robot) = i;
        pruned_idx_robot = pruned_idx_robot + 1;
    end
end
if(isempty(pruned_entries_robot)==0)
    robot_data = removerows(robot_data, 'ind', pruned_entries_robot);
end

% select and remove out-of-range entries for camera sensory dataset
pruned_entries_vision = [];
pruned_idx_vision = 1;
for i=1:length(vision_data)
    % start check
    if (vision_data(i,1) - global_start < tolerance)
        pruned_entries_vision(pruned_idx_vision) = i;
        pruned_idx_vision = pruned_idx_vision + 1;
    end
    % end check
    if (vision_data(i,1) - global_stop >  tolerance)
        pruned_entries_vision(pruned_idx_vision) = i;
        pruned_idx_vision = pruned_idx_vision + 1;
    end
end
if(isempty(pruned_entries_vision)==0)
    vision_data = removerows(vision_data, 'ind', pruned_entries_vision);
end

% check the sizes after pruning the out-of-bound entries
if (length(robot_data) == length(vision_data))
    % go on with analysis
else
    pruning_cand_robot = [];
    pruning_cand_robot_idx = 1;
    pruning_cand_vision = [];
    pruning_cand_vision_idx = 1; 
    % check which dataset is longer and check the timestamp diff
    if (length(robot_data) > length(vision_data))
        % robot dataset is bigger so shrink it
        % compare the beggining and the end of the vision dataset until we
        % reach same size
        for i=1:length(robot_data)
            if(robot_data(i,1) - vision_data(1,1) <  tolerance)
                pruning_cand_robot(pruning_cand_robot_idx) = i;
                pruning_cand_robot_idx = pruning_cand_robot_idx + 1;
            end
            if(robot_data(i,1) - vision_data(length(vision_data),1) > tolerance)
                pruning_cand_robot(pruning_cand_robot_idx) = i;
                pruning_cand_robot_idx = pruning_cand_robot_idx + 1;
            end
        end
    else
        % vision dataset is bigger so shrink it
        % compare the beggining and the end of the vision dataset until we
        % reach same size
        for i=1:length(vision_data)
            if(vision_data(i,1) - robot_data(1,1) <  tolerance)
                pruning_cand_vision(pruning_cand_vision_idx) = i;
                pruning_cand_vision_idx = pruning_cand_vision_idx + 1;
            end
            if(vision_data(i,1) - robot_data(length(robot_data),1) > tolerance)
                pruning_cand_vision(pruning_cand_vision_idx) = i;
                pruning_cand_vision_idx = pruning_cand_vision_idx + 1;
            end
        end
    end
    robot_data = removerows(robot_data, 'ind', pruning_cand_robot);
    vision_data = removerows(vision_data, 'ind', pruning_cand_vision);
    
    % final cuts to fit the size
    diff_rt = length(robot_data) - length(vision_data);
    diff_tr = length(vision_data) - length(robot_data);
    if(sign(diff_rt)==1)
        for i=1:diff_rt
            robot_data = removerows(robot_data, 'ind', length(robot_data)-i);
        end
    end
    if(sign(diff_tr)==1)
        for i=1:diff_tr
            vision_data = removerows(vision_data, 'ind', length(vision_data)-i);
        end
    end
    
end

% input size (4 - modalities: gyro, compass, odometry, vision)
size_x = 4;
len_x  = length(robot_data);

% TODO input vector with samples from each modality
% preallocate
x = zeros(size_x, len_x);

% populate input vector from dataset

% raw gyro
x(1, :) = robot_data(:, 7); 

% raw compass
% adjust the sign of the magneto to be compliant with the other sensors
wrap_up_compass = 180;          % deg
robot_data(:,14) = -(robot_data(:,14));
magneto_scaling_factor = 1000;  % samples
magneto_raw    = robot_data(:,14)/magneto_scaling_factor;
magneto_offset = magneto_raw(1);
magneto_aligned = magneto_raw(:) - magneto_offset;
% handle jumps and wrap-ups
for i=2:length(magneto_aligned)
    delta = magneto_aligned(i-1) - magneto_aligned(i);
    if(delta > wrap_up_compass)
        magneto_aligned(i) = magneto_aligned(i) + 2*wrap_up_compass;
    end

end
head_magneto = magneto_aligned;
% populate the input vector entry
x(2, :) = head_magneto;

% raw odometry (needs transf. from wheel velo to change in angle)
% compute change in heading angle from odometry
dhead_odometry = zeros(1, length(robot_data));
omnirob_buffer_size = 8; % samples
robot_wheel_radius = 0.027; % m
robot_base = 0.082; %m
imu_scaling_factor = 1000; % samples
for i=1:length(robot_data)
    dhead_odometry(i) = (((pi/30) * (robot_data(i,2)/omnirob_buffer_size + ...
        robot_data(i,3)/omnirob_buffer_size + ...
        robot_data(i,4)/omnirob_buffer_size)*robot_wheel_radius)/...
        (3*robot_base))*(180/pi);
end
% populate the input vector entry
x(3, :) = dhead_odometry;

% raw vision 
% clean vision noise at startup (spurios jumps of the tracking software)
movement_limit_vision = 350;
for i=2:length(vision_data)
    delta = vision_data(i,4) - vision_data(i-1, 4);
    if(delta >= movement_limit_vision)
        vision_data(i,4) = vision_data(i-1,4);
    end
end
% compute vision offset
vision_offset = vision_data(2,4);
%remove vision offset
head_vision = vision_data(:,4) - vision_offset;
% populate the input vector 
x(4, :) = head_vision;

%% Network setup

% net params for structure and learning
net_size = 5; % neurons per lattice coordinate
gama0 = 0.1; % learning rate init
sigma0 = net_size/10; % neighborhood radius init
falpha  = 3; % neuron output activation function param 1
fbeta = 0.07; % neuron output activation function param 2
net_epochs = 10; % epochs to present data to the net
som_net_iter = 1; % current iteration in the input dataset
som_net_epoch = 1; % current training epoch

% net structure
som_neuron(1:net_size, 1:net_size) = struct('xpos', 0,...
                                            'ypos', 0,...
                                            'Wx'  , zeros(1, size_x),...
                                            'Wy'  , 0.0,...
                                            'y'   , 0.0);
for idx = 1:net_size
    for jdx = 1:net_size
        som_neuron(idx, jdx).xpos    = idx;
        som_neuron(idx, jdx).ypos    = jdx;
        for in_idx = 1:size_x
            som_neuron(idx, jdx).Wx(in_idx) = rand;
        end
        som_neuron(idx, jdx).Wy      = rand;
        som_neuron(idx, jdx).y       = 0.0;
    end
end

% the direct SOM
som_net = struct('units', som_neuron);   

% the feedback SOM
som_net_feedback = struct('units', som_neuron);   

% learning phase 
while(1)
    if(som_net_epoch < net_epochs)
        % if we are still training the net and didn't reach the max epoch
        % present input data vector to the network
        % for each entry in the input data vector
        for data_idx = 1:length(x)
           % init BMU
           bmu.xpos = 0;
           bmu.ypos = 0;
           bmu.Wx = rand(1, size_x);
           bmu.Wy = rand;
           bmu.y  = 0.0;
           % init max quantization error
           ei_max = bitmax;
           
           % search for the BMU
           for idx = 1:net_size
               for jdx = 1:net_size
                    % compute the quantization error between current sensory input
                    % vector, old value input and the weights
                    WX_VECT = som_net.units;
                    WX1 = WX_VECT(idx, jdx).Wx(1);
                    WX2 = WX_VECT(idx, jdx).Wx(2);
                    WX3 = WX_VECT(idx, jdx).Wx(3);
                    WX4 = WX_VECT(idx, jdx).Wx(4);
                    WY  = WX_VECT(idx, jdx).Wy;
                    Y_VECT_FDBK = som_net_feedback.units;
                    Y_VECT = Y_VECT_FDBK(idx, jdx).y;
                    ei = falpha*norm([ x(1, data_idx) - WX1,...
                                       x(2, data_idx) - WX2,...
                                       x(3, data_idx) - WX3,...
                                       x(4, data_idx) - WX4], 2)^2+...
                         fbeta*norm(Y_VECT - WY, 2)^2;
                    
                    % compute the activation (output) of the current neuron
                    som_net(idx, jdx).y = exp(ei);
                    
                    % check if the current neuron is the BMU
                    if(ei < ei_max)
                        % if bmu found
                        bmu.xpos = idx;
                        bmu.ypos = jdx;
                        ei_max = ei;
                    end
               end
           end
           
           % adjust the neighborhood kernel 
           for idx = 1:net_size
               for jdx = 1:net_size
                    % time ct in radius and learning rate adaptation
                    lambda = net_epochs/log(sigma0);
                    % update the learning rate 
                    gama = gama0 * exp(-(som_net_iter)/lambda);
                    % update the neighborhood radius size
                    sigma = sigma0*exp(-som_net_iter/lambda);
                    % compute the value of the neighborhood kernel
                    h = gama*exp(-(sqrt((bmu.xpos - idx)^2+(bmu.ypos - jdx)^2))^2/(2*sigma^2));
                    
                    % update weights for the current neuron in the BMU
                    % neighborhood if in the radius
                    if(sqrt((bmu.xpos - idx)^2+(bmu.ypos - jdx)^2) < sigma^2)
                        for in_idx = 1:size_x
                            WX_VECT_FIELD = som_net.units;
                            WX_VECT = WX_VECT_FIELD(idx, jdx);
                            WX = WX_VECT.Wx(in_idx);
                            WX =  WX + h*(bmu.Wx(in_idx) - WX);
                            WX = abs(WX);
                        end
                        WY_VECT_FIELD = som_net.units;
                        WY_VECT = WY_VECT_FIELD(idx, jdx);
                        WY = WY_VECT.Wy;
                        WY =  WY + h*(bmu.Wy - WY);
                        WY = abs(WY);
                    end
               end
           end
           
           % increment iteration through input vector 
           som_net_iter = som_net_iter + 1;
           if(som_net_iter==len_x)
               som_net_iter = 1;
               fprintf(1, sprintf('Training epoch %d\n', som_net_epoch));
               som_net_epoch = som_net_epoch + 1;
           end
           % copy the current activity in the direct SOM in the feedback SOM
           som_net_feedback = som_net;
        end
    else
       disp 'Finalized learning phase';
       break; 
    end
end


                               