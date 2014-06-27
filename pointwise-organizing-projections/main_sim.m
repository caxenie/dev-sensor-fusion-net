%% Pointwise Organizing Projecitons - Self-Organizing-Maps for pointwise projections - Kohonen, 2005

% environment setup
clear all; clc; close all;
%% INITIALIZATION
% DATA
IN_TYPE = 'gauss-mixture';  % input signal type
DATASET_SIZE = 1;         % size of the input dataset (number of input vector)
K = 3;                      % number of Gaussians in the mixture
SIGMA0 = 5;                  % standard deviation of each Gaussian
% NETWORK
IN_SIZE = 50;              % neurons in the input layer
OUT_SIZE = 50;             % neurons in the output layer
MAX_EPOCHS = 125000;         % epochs to train the network
A = 4;                      % coef in the nonlinear effect of plasticity control
C = 5;                      % interaction range parameter
BETA = 1000;                % relative strength of sysnapse-dependent impact in learning
% NETWORK STRUCTURE
net.iter = 1;
net.maxiter = MAX_EPOCHS;
net.insize = IN_SIZE;
net.outsize = OUT_SIZE;
net.w = zeros(net.outsize, net.insize, net.maxiter);
for tidx = 1:net.maxiter
    net.w(:, :, tidx) = rand(net.outsize, net.insize);                % synaptic weights
end
net.x = zeros(net.maxiter, net.insize);                           % input layer activity (presynaptic)
net.y = zeros(net.maxiter, net.outsize);                          % transfer function
net.z = zeros(net.maxiter, net.outsize);                          % spreading plasticity control
net.u = zeros(net.maxiter, net.outsize);                          % nonliearity in plasticity control (neigh function)

% initial random weights
W0 = net.w(:,:,1);
% learning rate init
alpha = zeros(1, net.maxiter);

% build the input signal in the network
% as a mixture of k Gaussians
for vidx = 1:DATASET_SIZE
    in_dataset(vidx) = generate_input_dataset(K, SIGMA0, IN_SIZE, IN_TYPE);
end

%% NETWORK TRAINING
while(net.iter <= MAX_EPOCHS)
    
    % compute the learning rate @ current training epoch with an inverse
    % time law
    alpha(net.iter) = 0.2/(1 + 0.001*net.iter);
    
    % select vector from the input dataset
    for inv_idx = 1:DATASET_SIZE
        % extract the date from dataset
        net.x(net.iter, :) = in_dataset(inv_idx).data;
        
        % compute the superpositive transfer function
        for idx = 1:net.outsize
            sumin = 0.0;
            for jdx = 1:net.insize
                sumin = sumin + net.w(idx, jdx, net.iter)*net.x(net.iter, jdx);
            end
            net.y(net.iter, idx) = sumin;
        end
        % normalize with respect to the max value
        net.y(net.iter,:) = net.y(net.iter, :)./max(net.y(net.iter, :));
                
        % compute the laterally spreading plasticity control agent
        for idx = 1:net.outsize
            sumout = 0.0;
            for hdx = 1:net.outsize
                % compute the one dimensional kernel
                gih = 1/(1+(norm(idx-hdx))/C);
                sumout = sumout + gih*net.y(net.iter, hdx);
            end
            net.z(net.iter, idx) = sumout;
        end
        % normalize with respect to the max value
        net.z(net.iter,:) = net.z(net.iter, :)./max(net.z(net.iter, :));
        
        
        % compute the nonliearity for the plasticity control as an exponential
        for idx = 1:net.outsize
            net.u(net.iter, idx) = exp(A*net.z(net.iter, idx));
        end
        % normalize with respect to the sum of elements
        net.u(net.iter, :) = net.u(net.iter, :)./sum(net.u(net.iter, :));
        
        % update the weights
        for idx = 1:net.outsize
            for jdx = 1:net.insize
                net.w(idx, jdx, net.iter) = net.w(idx, jdx, net.iter)+alpha(net.iter)*(1 + BETA*net.w(idx, jdx, net.iter)*net.u(net.iter, idx)*net.x(net.iter, jdx));
            end
        end
        % normalize using Euclidian norm
        normalization_factor = norm(net.w(:, :, net.iter));
        net.w(:, :, net.iter) = net.w(:, :, net.iter)./normalization_factor;
    end
    % incrment epoch counter
    net.iter = net.iter + 1;
end

%% VISUALIZATION
% plot the weights before and after training to check for self organization
% in the representation from the input layer to the output layer
figure; set(gcf, 'color', 'white'); grid off;
subplot(1, 3, 1);
plot(in_dataset.data);
title('Input signal: mixture of Gaussians');
subplot(1, 3, 2);
init_w = W0;
imagesc(init_w(1:net.outsize, 1:net.insize)); colormap(flipud(gray(256))); colorbar; box off; 
title('Initial connectivity: random synaptic weights');
subplot(1, 3, 3);
end_w = net.w(:, :, end);
imagesc(end_w(1:net.outsize, 1:net.insize)); colormap(flipud(gray(256))); colorbar; box off;
title('Final connectivity: ordered synaptic weights');