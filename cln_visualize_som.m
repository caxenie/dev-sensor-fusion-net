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
% check if cross modal interaction is activated
if(strcmp(visin.simopts.net.xmodlearn, 'none')~=1)
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
    % if we have a 2D SOM use 3D display for activities
    if(visin.simopts.net.sizex~=1)
        subplot(1,2,1);
        surf(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey));
        axis xy; caxis([0.0, 1.0]);
        xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
        subplot(1,2,2);
        imagesc(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey));
        colormap; colorbar; caxis([0.0, 1.0]);
        fig_title = sprintf('Total (joint) activity in network %s', curr_somid); suptitle(fig_title);
        axis xy; xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
    else
        plot(1:visin.simopts.net.sizey, visual_som, 'k', 'LineWidth', 4);
        fig_title = sprintf('Total (joint) activity in network %s', curr_somid); suptitle(fig_title);
        axis xy; xlabel('Neuron index'); ylabel('Activity'); box off;
    end
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
    % if we have a 2D SOM use 3D display for activities
    if(visin.simopts.net.sizex~=1)
        subplot(1,2,1);
        surf(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); caxis([0.0, 1.0]);
        axis xy; xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
        subplot(1,2,2);
        imagesc(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); colormap; colorbar; axis xy;
        fig_title = sprintf('Cross-modal elicited act. in network %s', curr_somid);suptitle(fig_title);
        xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
    else
        plot(1:visin.simopts.net.sizey, visual_som, 'k', 'LineWidth', 4);
        fig_title = sprintf('Cross-modal elicited act. in network %s', curr_somid); suptitle(fig_title);
        axis xy; xlabel('Neuron index'); box off;
    end
end % end check if cross modal interaction is enabled
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
% if we have a 2D SOM use 3D display for activities
if(visin.simopts.net.sizex~=1)
    subplot(1,2,1);
    surf(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); caxis([0.0, 1.0]);
    axis xy; xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
    subplot(1,2,2);
    imagesc(visual_som(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey));
    colormap; colorbar; axis xy; caxis([0.0, 1.0]);
    fig_title = sprintf('Sensory elicited act. in network %s', curr_somid); suptitle(fig_title);
    axis xy; xlabel('Neuron index'); ylabel('Neuron index'); zlabel('Activity');
else
    plot(1:visin.simopts.net.sizey, visual_som, 'k', 'LineWidth', 4);
    fig_title = sprintf('Sensory elicited activity in network %s', curr_somid); suptitle(fig_title);
    axis xy; xlabel('Neuron index'); ylabel('Activity'); box off;
end
%----------------------------------------------------------------
% synaptic connections strenghts from sensory projections (W weight matrix)
% present data only if is not in hunt mode
if(strcmp(visin.simopts.data.trainvtype, 'hunt')==0)
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
                mean_qe(indx) = mean(som(cidx, ridx).W(indx), 1);
                stdev(indx) = std(visin.netin.trainv1(:, indx) - som(cidx, ridx).W(indx), 0, 1);
            end
        case '2'
            for indx = 1:visin.simopts.data.trainvsize
                mean_qe(indx) = mean(som(cidx, ridx).W(indx), 1);
                stdev(indx) = std(visin.netin.trainv2(:, indx) - som(cidx, ridx).W(indx), 0, 1);
            end
    end
    % plot the quantization error (+/- standard deviation) overlayed on the preferred value of each
    % neuron after training
    samples = 1:visin.simopts.data.trainvsize;
    plus_sd = [mean_qe + stdev];  
    minus_sd = [mean_qe - stdev];
    patch([samples, samples(end:-1:1)],...
           [minus_sd, plus_sd(end:-1:1)],...
           [0.7 0.65 0.71]); hold on; box off;
    % overlay weitght vector for current neuron
    plot(som(cidx, ridx).W);
end
fig_title = sprintf('Sensory projections synaptic weights in network %s', curr_somid);suptitle(fig_title);
end
%----------------------------------------------------------------
% component planes analysis for the correlation hunting train type
if(strcmp(visin.simopts.data.trainvtype, 'hunt')==1)
    % first find the bounds in the data so that all colormaps are aligned 
    minmax = zeros(2, visin.simopts.data.trainvsize);
    for idx = 1:visin.simopts.data.trainvsize
    Wshow = zeros(visin.simopts.net.sizex, visin.simopts.net.sizey);
        for xidx = 1:visin.simopts.net.sizex
            for yidx = 1:visin.simopts.net.sizey
                Wshow(xidx, yidx) = som(xidx, yidx).W(idx); 
            end
        end
        minmax(1, idx) = min(Wshow(:));
        minmax(2, idx) = max(Wshow(:));
    end
    switch curr_somid
        case '1'
             min_scale = min(visin.netin.trainv1(:, 1));  % min(minmax(1,:));
             max_scale = max(visin.netin.trainv1(:, 1)); % max(minmax(2,:));
        case '2'
             min_scale = min(visin.netin.trainv2(:, 1));  % min(minmax(1,:));
             max_scale = max(visin.netin.trainv2(:, 1)); % max(minmax(2,:));
    end
    figure; 
    set(gcf, 'color', 'white'); box off; grid off;
    % plot each component from the input vector for each neuron in the som
    % slice the som so that component planes emerge
    for idx = 1:visin.simopts.data.trainvsize
    subplot(1, visin.simopts.data.trainvsize, idx);
    Wshow = zeros(visin.simopts.net.sizex, visin.simopts.net.sizey);
        for xidx = 1:visin.simopts.net.sizex
            for yidx = 1:visin.simopts.net.sizey
                Wshow(xidx, yidx) = som(xidx, yidx).W(idx); 
            end
        end
        % plot the weights
        imagesc(Wshow(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey)); %, [min_scale, max_scale]);
        colorbar; axis xy; colormap('gray'); box off; 
    end
    fig_title = sprintf('Component planes for net %s (colormap)', curr_somid);suptitle(fig_title);
    
    % add a view of the data representing each component from the training
    % vectors which are learned by the network
    figure;
    set(gcf, 'color', 'white'); grid off;
    for idx = 1:visin.simopts.data.trainvsize
        subplot(1, visin.simopts.data.trainvsize, idx);
        switch(curr_somid)
            case '1'
                plot(visin.netin.trainv1(:,idx)); box off;
            case '2'
                plot(visin.netin.trainv2(:,idx)); box off;
        end
        xlabel('Samples'); ylabel(sprintf('Component %d', idx)); 
    end
    
    % plot the learned dependencies in the network
    % plot the independent component planes values one against each other
    % and compare with input data to check the quantization error
    % analyze each component
    figure;
    set(gcf, 'color', 'white'); grid off;
    for idx = 1:visin.simopts.data.trainvsize
        subplot(1, visin.simopts.data.trainvsize, idx);
        switch(curr_somid)
            case '1'
                plot(visin.netin.trainv1(:,idx), 'k'); hold on;
                plot(visin.bmu1_hist(idx, :), 'g'); box off;
                xlabel('Samples'); legend('Input data','Learned data'); 
            case '2'
                plot(visin.netin.trainv2(:,idx),'k'); hold on;
                plot(visin.bmu2_hist(idx, :), 'g'); box off;
                xlabel('Samples'); legend('Input data','Learned data'); 
        end
    end
    suptitle('Per component analysis');
    
    % plot the learned dependencies in the network
    % plot the independent component planes values one against each other
    % and compare with input data to check the quantization error
    % analyze all components 
    figure;
    set(gcf, 'color', 'white'); grid off;
        switch(curr_somid)
            case '1'
                plot(visin.netin.trainv1(end,:), 'k'); hold on;
                plot(visin.bmu1_hist(:, end), 'g'); box off;
                legend('Input data','Learned data'); 
            case '2'
                plot(visin.netin.trainv2(end,:), 'k'); hold on;
                plot(visin.bmu2_hist(:, end), 'g'); box off;
                legend('Input data','Learned data'); 
        end
    suptitle('All components analysis');
    
    % plot the dependencies as learned from the net and the input data fed
    % to the network for training
    figure;
    set(gcf, 'color', 'white'); grid off;
    for idx = 1:visin.simopts.data.trainvsize
        subplot(1, visin.simopts.data.trainvsize, idx);
        switch(curr_somid)
            case '1'
                plot(visin.netin.trainv1(:,1), visin.netin.trainv1(:,idx), 'k'); hold on;
                plot(visin.bmu1_hist(1, :), visin.bmu1_hist(idx, :), 'g'); box off;
                legend('Input data','Learned data'); 
            case '2'
                plot(visin.netin.trainv2(:,1), visin.netin.trainv2(:,idx), 'k'); hold on;
                plot(visin.bmu2_hist(1, :), visin.bmu2_hist(idx, :), 'g'); box off;
                legend('Input data','Learned data'); 
        end
    end
    suptitle('Correlation plots');
      
else
    
% synaptic connections strenghts in color map in neuron - component view
% for 1D som
figure;
set(gcf, 'color', 'white'); box off; grid off;
szx = visin.simopts.net.sizey;
szy = visin.simopts.data.trainvsize;
Wshow = zeros(szx, szy);
for sidx = 1:szx
    for tidx = 1:szy
        Wshow(sidx, tidx) = som(sidx).W(tidx);
    end
end
% plot the weights
imagesc(Wshow(1:szx, 1:szy)');
colorbar; axis xy; colormap; box off;
fig_title = sprintf('Sensory projections synaptic weights in network %s (colormap)', curr_somid);suptitle(fig_title);
end

% check if cross-modal interaction is enabled
if(strcmp(visin.simopts.net.xmodlearn, 'none')~=1)
    %----------------------------------------------------------------
    % synaptic connections strenghts from cross modal Hebbian interaction (H weight matrix)
    figure;
    set(gcf, 'color', 'white'); box off; grid off;
    coln = visin.simopts.net.sizey; % true for square matrix
    rown = visin.simopts.net.sizex;
    % make sure that the cross modal weights suplot agrees with the network
    % layout for neurons as used in activity and component planes analysis
    subplot_align_raw = reshape(1:coln*rown, coln, [])';
    subplot_align_prep = flipud(subplot_align_raw);
    subplot_align = reshape(subplot_align_prep.',1,[]);
    for sidx = 1:rown*coln
        subplot(rown, coln, subplot_align(sidx));
        [ridx, cidx] = ind2sub([coln, rown], sidx);
        % plot the weights for current neuron
        imagesc((som(cidx, ridx).H(1:visin.simopts.net.sizex, 1:visin.simopts.net.sizey))); hold on;
        %if(ridx == visin.simopts.net.size)
        colorbar; caxis([0.0 1.0]);
        %end
        axis xy; colormap; box off; caxis([0.0 1.0]);
    end
    fig_title = sprintf('Cross-modal synaptic weights in network %s', curr_somid);suptitle(fig_title);
end
end
