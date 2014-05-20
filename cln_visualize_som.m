%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% For the current implementation we only consider a 2 variable network

% FUNCTION
% Visualize SOM specific data (adaptive params, activations, weights).
%
% A
%   som - network to visualize
%
% RETURN
%   somfig - figures handles for later access

function somfig = cln_visualize_som(visin, som)
% get the id of the current som
somid = [som.id]; curr_somid = somid(end);
% total activity in each neurons in the SOM
figure;
set(gcf, 'color', 'white'); box off; grid off;
% visualization of direct activity
visual_som = zeros(visin.simopts.net.size, visin.simopts.net.size);
for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        visual_som(idx, jdx) = som(idx, jdx).at;
    end
end
subplot(1,2,1); surf(visual_som(1:visin.simopts.net.size, 1:visin.simopts.net.size)); axis xy;
subplot(1,2,2); imagesc(visual_som(1:visin.simopts.net.size, 1:visin.simopts.net.size)); colormap; colorbar; axis xy;
fig_title = sprintf('Total (joint) activity in network %s', curr_somid); suptitle(fig_title);

figure;
set(gcf, 'color', 'white'); box off; grid off;
% visualization of direct activity
visual_som = zeros(visin.simopts.net.size, visin.simopts.net.size);
for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        visual_som(idx, jdx) = som(idx, jdx).ad;
    end
end
subplot(1,2,1); surf(visual_som(1:visin.simopts.net.size, 1:visin.simopts.net.size)); axis xy;
subplot(1,2,2); imagesc(visual_som(1:visin.simopts.net.size, 1:visin.simopts.net.size)); colormap; colorbar; axis xy;
fig_title = sprintf('Sensory elicited act. in network %s', curr_somid); suptitle(fig_title);

% indirect activity elicited by cross-modal Hebbian linkage (plastic connections)
figure;
set(gcf, 'color', 'white'); box off; grid off;
% visualization of indirect activity (cross-modal elicited activity)
visual_som = zeros(visin.simopts.net.size, visin.simopts.net.size);
for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        visual_som(idx, jdx) = som(idx, jdx).ai;
    end
end
subplot(1,2,1); surf(visual_som(1:visin.simopts.net.size, 1:visin.simopts.net.size)); axis xy;
subplot(1,2,2); imagesc(visual_som(1:visin.simopts.net.size, 1:visin.simopts.net.size)); colormap; colorbar; axis xy;
fig_title = sprintf('Cross-modal elicited act. in network %s', curr_somid);suptitle(fig_title);

% synaptic connections strenghts from sensory projections (W weight matrix)
figure;
set(gcf, 'color', 'white'); box off; grid off;
% present each neurons receptive field (mean input sequence that triggers that neuron - weight vector)
coln = visin.simopts.net.size; % true for square matrix
rown = visin.simopts.net.size;
for sidx = 1:rown*coln
    subplot(rown, coln, sidx);
    [ridx, cidx] = ind2sub([coln, rown], sidx);
    plot(som(cidx, ridx).W); box off; axis([1 length(visin.netin.trainv2) min(min([som.W])) max(max([som.W]))]); hold on; 
    switch curr_somid
        case '1'
            plot(visin.netin.trainv1(end,:), 'r');
        case '2'
            plot(visin.netin.trainv2(end,:), 'm');
    end
end
fig_title = sprintf('Sensory projections synaptic weights in network %s', curr_somid);suptitle(fig_title);

% synaptic connections strenghts from cross modal Hebbian interaction (H weight matrix)
figure;
set(gcf, 'color', 'white'); box off; grid off;
coln = visin.simopts.net.size; % true for square matrix
rown = visin.simopts.net.size;
for sidx = 1:rown*coln
    subplot(rown, coln, sidx);
    [ridx, cidx] = ind2sub([coln, rown], sidx);
    % plot the weights for current neuron
    imagesc(som(cidx, ridx).H(1:visin.simopts.net.size, 1:visin.simopts.net.size)); hold on; 
    %if(ridx == visin.simopts.net.size) 
    colorbar; 
    %end
    axis xy; colormap; box off;
end
fig_title = sprintf('Cross-modal synaptic weights in network %s', curr_somid);suptitle(fig_title);
somfig = 1; % fixme - return all handles when visualization is ready
end

