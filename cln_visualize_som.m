%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% For the current implementation we only consider a 2 variable network

% FUNCTION
% Visualize SOM specific data (adaptive params, activations, weights).
%
% ARGS
%   som - network to visualize
%
% RETURN
%   void

function cln_visualize_som(visin, som)
% get the id of the current som
somid = [som.id]; curr_somid = somid(end);
%----------------------------------------------------------------
% total activity in each neurons in the SOM
figure; set(gcf, 'color', 'white'); box off; grid off;
% visualization of direct activity
visual_som = zeros(visin.simopts.net.sizex, visin.simopts.net.sizey);
for idx = 1:visin.simopts.net.sizex
    for jdx = 1:visin.simopts.net.sizey
        visual_som(idx, jdx) = som(idx, jdx).at;
    end
end
subplot(1,2,1); 
surf(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); 
axis xy; caxis([0.0, 1.0]);
xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
subplot(1,2,2); 
imagesc(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); 
colormap; colorbar; caxis([0.0, 1.0]);
fig_title = sprintf('Total (joint) activity in network %s', curr_somid); suptitle(fig_title); 
axis xy; xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
%----------------------------------------------------------------
figure;
set(gcf, 'color', 'white'); 
box off; grid off;
% visualization of direct activity
visual_som = zeros(visin.simopts.net.sizex, visin.simopts.net.sizey);
for idx = 1:visin.simopts.net.sizex
    for jdx = 1:visin.simopts.net.sizey
        visual_som(idx, jdx) = som(idx, jdx).ad;
    end
end
subplot(1,2,1); 
surf(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); caxis([0.0, 1.0]);
axis xy; xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
subplot(1,2,2); 
imagesc(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); 
colormap; colorbar; axis xy; caxis([0.0, 1.0]);
fig_title = sprintf('Sensory elicited act. in network %s', curr_somid); suptitle(fig_title); 
axis xy; xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
%----------------------------------------------------------------
% indirect activity elicited by cross-modal Hebbian linkage (plastic connections)
figure;
set(gcf, 'color', 'white'); box off; grid off;
% visualization of indirect activity (cross-modal elicited activity)
visual_som = zeros(visin.simopts.net.sizex, visin.simopts.net.sizey);
for idx = 1:visin.simopts.net.sizex
    for jdx = 1:visin.simopts.net.sizey
        visual_som(idx, jdx) = som(idx, jdx).ai;
    end
end
subplot(1,2,1); 
surf(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); caxis([0.0, 1.0]);
axis xy; xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
subplot(1,2,2); 
imagesc(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); colormap; colorbar; axis xy; 
fig_title = sprintf('Cross-modal elicited act. in network %s', curr_somid);suptitle(fig_title);
xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
%----------------------------------------------------------------
% synaptic connections strenghts from sensory projections (W weight matrix)
figure;
set(gcf, 'color', 'white'); box off; grid off;
% present each neurons receptive field (mean input sequence that triggers that neuron - weight vector)
coln = visin.simopts.net.sizey; % true for square matrix
rown = visin.simopts.net.sizex;
mean_qe = zeros(1, visin.simopts.data.trainvsize);
stdev = zeros(1, visin.simopts.data.trainvsize);
for sidx = 1:rown*coln
    subplot(rown, coln, sidx);
    [ridx, cidx] = ind2sub([coln, rown], sidx);
    switch curr_somid
        case '1'
            % compute the mean quantization error between all input vectors
            % from the training dataset and the current neuron receptive
            % field (prefered value)
            for indx = 1:visin.simopts.data.trainvsize
                mean_qe(indx) = mean(visin.netin.trainv1(:, indx) - som(cidx, ridx).W(indx));
                stdev(indx) = std(visin.netin.trainv1(:, indx) - som(cidx, ridx).W(indx));
            end
        case '2'
            for indx = 1:visin.simopts.data.trainvsize
                mean_qe(indx) = mean(visin.netin.trainv2(:, indx) - som(cidx, ridx).W(indx));
                stdev(indx) = std(visin.netin.trainv2(:, indx) - som(cidx, ridx).W(indx));
            end
    end
    % plot the quantization error (+/- standard deviation) overlayed on the preferred value of each
    % neuron after training
    sample_index = 1:visin.simopts.data.trainvsize;
    plus_sd = [sample_index, fliplr(sample_index)];
    minus_sd = [som(cidx, ridx).W + stdev, fliplr(som(cidx, ridx).W-stdev)];
    fill(plus_sd, minus_sd, 'y'); hold on; box off;     
    % overlay weitght vector for current neuron
    plot(som(cidx, ridx).W);
end
fig_title = sprintf('Sensory projections synaptic weights in network %s', curr_somid);suptitle(fig_title);
%----------------------------------------------------------------
% synaptic connections strenghts from cross modal Hebbian interaction (H weight matrix)
figure;
set(gcf, 'color', 'white'); box off; grid off;
coln = visin.simopts.net.sizey; % true for square matrix
rown = visin.simopts.net.sizex;
for sidx = 1:rown*coln
    subplot(rown, coln, sidx);
    [ridx, cidx] = ind2sub([coln, rown], sidx);
    % plot the weights for current neuron
    imagesc(som(cidx, ridx).H(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); hold on; 
    %if(ridx == visin.simopts.net.size) 
    colorbar; caxis([0.0 1.0]);
    %end
    axis xy; colormap; box off; caxis([0.0 1.0]);
end
fig_title = sprintf('Cross-modal synaptic weights in network %s', curr_somid);suptitle(fig_title);
end

