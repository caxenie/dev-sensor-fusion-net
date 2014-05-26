%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% For the current implementation we only consider a 2 variable network

% FUNCTION
% Run the main network dynamics loop.
%
% ARGS
%   simopts - simulation options
%   netin   - sensory data
%   som1    - first som net
%   som2    - second som net
%
% RETURN
%   rundata - file name (runtime data dumped to a file for analysis)

function rundata = cln_iterate_network(simopts, netin, som1, som2)
% network iterator
net_iter = 1;
% quantization error for each map and intermap
qedir1     = zeros(simopts.net.size, simopts.net.size);
qedir2     = zeros(simopts.net.size, simopts.net.size);
cross_mod1 = zeros(simopts.net.size, simopts.net.size);
cross_mod2 = zeros(simopts.net.size, simopts.net.size);
% init adaptive params
alphat = zeros(1, simopts.net.maxepochs);
sigmat = zeros(1, simopts.net.maxepochs);
gammat = zeros(1, simopts.net.maxepochs);
xit    = zeros(1, simopts.net.maxepochs);
kappat = zeros(1, simopts.net.maxepochs);
% set up the initial values
alphat(1) = simopts.net.alpha;
sigmat(1) = simopts.net.sigma;
gammat(1) = simopts.net.gamma;
xit(1)    = simopts.net.xi;
kappat(1) = simopts.net.kappa;
tau = 500;
if(simopts.debug.visual==1)
    % epoch wise visualization of network activities(direct, indirect and total)
    act_vis = figure; set(gcf, 'color', 'white'); box off;
    visual_somd1 = zeros(simopts.net.size, simopts.net.size);
    visual_somd2 = zeros(simopts.net.size, simopts.net.size);
    visual_somi1 = zeros(simopts.net.size, simopts.net.size);
    visual_somi2 = zeros(simopts.net.size, simopts.net.size);
    visual_somt1 = zeros(simopts.net.size, simopts.net.size);
    visual_somt2 = zeros(simopts.net.size, simopts.net.size);
    hweight_vis1 = figure; set(gcf, 'color', 'white'); box off;
    hweight_vis2 = figure; set(gcf, 'color', 'white'); box off;
    coln = simopts.net.size; % true for square matrix
    rown = simopts.net.size;
    figure; set(gcf, 'color', 'white');
    plot(netin.trainv1(1, :),'r'); hold on; plot(netin.trainv2(1, :), 'b');
    legend('First input, p1', 'Second input, p2'); box off;
end
% main loop of the network
while(1)
    % check if we finished training
    if(net_iter <= simopts.net.maxepochs)
        
        % adaptive process parameters (learning rates, cross-modal
        % modulation factor etc.
        switch(simopts.net.params)
            case 'fixed'
                % -------------------------------------------------------------------------------
                % compute the learning rate @ current epoch
                alphat(net_iter) = alphat(1);
                % adapt the cross-modal interaction params (increase in time)
                gammat(net_iter) = gammat(1);
                % inhibitory component for co-activation
                xit(net_iter) = xit(1);
                % Hebbian learning rate
                kappat(net_iter) = kappat(1);
                % -------------------------------------------------------------------------------                
            case 'adaptive'
                
                % use the same learning parameters for both SOM
                
                % for thresholding of the adaptive parameters use
                %   param_out = min( max( param_in, min_value ), max_value );
                
                % the learning rate and radius decrease over time to
                % enable learning on a coarse and then on a finer time
                % scale
                
                % -------------------------------------------------------------------------------
                % compute the learning rate @ current epoch
                % exponential learning rate adaptation
                alphat(net_iter) = simopts.net.alpha*exp(-net_iter/tau);
                % semi-empirical learning rate adaptation - inverse
                % time adaptation
                % A = simopts.net.maxepochs/100.0; B = A;
                % alphat(net_iter) = A/(net_iter + B);
                % -------------------------------------------------------------------------------
                % adapt the cross-modal interaction params (increase in time)
                
                % cross-modal activation impact on local som learning
                gammat(net_iter) = simopts.net.gamma*exp(net_iter/tau);
                % inhibitory component to ensure only co-activation
                xit(net_iter) = simopts.net.xi*exp(net_iter/tau);
                % Hebbian learning rate
                kappat(net_iter) = simopts.net.kappa*exp(net_iter/tau);
                
                %----------------------------------------------------------------------------------------------------------
        end % end switch adaptive process parameteres type selection
        
        % ---------------------------------------------------------------------------------------------------------
        % compute the neighborhood radius size @ current epoch
        sigmat(net_iter) = simopts.net.sigma*exp(-net_iter/simopts.net.lambda);
        %----------------------------------------------------------------------------------------------------------
        
        % present a vector from each training data set to the network's
        % SOMs - afferent projections from sensors
        
        for trainv_idx = 1:netin.trainsetsize
            % max quantization error init
            qe_max1_dir = Inf; qe_max2_dir = Inf;
            % go through the two SOMs
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                    % compute quantization errors in each som map for
                    % afferent projections from the sensors
                    qedir1(idx, jdx) = norm(netin.trainv1(trainv_idx, :) - som1(idx, jdx).W);
                    % check if current neuron is winner
                    if(qedir1(idx, jdx)<qe_max1_dir)
                        bmudir1.xpos = idx;
                        bmudir1.ypos = jdx;
                        bmudir1.qe = qedir1(idx, jdx);
                        qe_max1_dir = qedir1(idx, jdx);
                    end
                    qedir2(idx, jdx) = norm(netin.trainv2(trainv_idx, :) - som2(idx, jdx).W);
                    % check if current neuron is winner
                    if(qedir2(idx, jdx)<qe_max2_dir)
                        bmudir2.xpos = idx;
                        bmudir2.ypos = jdx;
                        bmudir2.qe = qedir2(idx, jdx);
                        qe_max2_dir = qedir2(idx, jdx);
                    end
                end
            end % end of bmus search loop
            % check if verbose is on for debugging
            if(simopts.debug.verbose == 1)
                % first som
                fprintf('-------------BMU Search Phase------------\n');
                fprintf(1, 'SOM1 \n IN_VEC:\n'); netin.trainv1(trainv_idx, :)
                fprintf(1, 'W_VEC_WIN(%d, %d) \n', bmudir1.xpos, bmudir1.ypos); som1(bmudir1.xpos, bmudir1.ypos).W
                fprintf(1, 'WIN_POS:(%d,%d) - WIN:%f\n',bmudir1.xpos, bmudir1.ypos, bmudir1.qe);
                fprintf(1, 'QE_MAT:\n'); qedir1
                % second som
                fprintf(1, 'SOM2 \n IN_VEC:\n'); netin.trainv2(trainv_idx, :)
                fprintf(1, 'W_VEC_WIN(%d, %d) \n', bmudir2.xpos, bmudir2.ypos); som2(bmudir2.xpos, bmudir2.ypos).W
                fprintf(1, 'WIN_POS:(%d,%d) - WIN:%f\n',bmudir2.xpos, bmudir2.ypos, bmudir2.qe);
                fprintf(1, 'QE_MAT:\n'); qedir2
            end
            
            % compute the activations for each neuron in each som
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                    %----------------------------------------------------------------------------------------------------------
                    
                    % compute the direct activation - neighborhood kernel
                    
                    % for SOM1
                    som1(idx, jdx).ad = exp(-(norm([bmudir1.xpos - idx, bmudir1.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)));
                    
                    % for SOM2
                    som2(idx, jdx).ad = exp(-(norm([bmudir2.xpos - idx, bmudir2.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)));
                end
            end
            % check if verbose is on for debugging
            if(simopts.debug.verbose == 1)
                fprintf('-------------Direct Activation Phase------------\n');
                % first som
                fprintf(1, 'SOM1BMU_DIR = (%d,%d) with ad = %f \n', bmudir1.xpos, bmudir1.ypos, som1(bmudir1.xpos, bmudir1.ypos).ad);
                fprintf(1, 'SOM1.ad = \n'); [som1.ad]
                % second som
                fprintf(1, 'SOM2BMU_DIR = (%d,%d) with ad = %f \n', bmudir2.xpos, bmudir2.ypos, som2(bmudir2.xpos, bmudir2.ypos).ad);
                fprintf(1, 'SOM2.ad = \n'); [som2.ad]
            end
            
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                    % ---------------------------------------------------------------------------------------------------------
                    % compute the indirect activation
                    
                    % from all other units in SOM2
                    for isom1 = 1:simopts.net.size
                        for jsom1 = 1:simopts.net.size
                            for isom2 = 1:simopts.net.size
                                for jsom2 = 1:simopts.net.size
                                    % sum of all products between direct activation
                                    % and cross-som Hebbian weights
                                    cross_mod2(isom1, jsom1) = cross_mod2(isom1, jsom1) + ...
                                        som2(isom2, jsom2).ad*som2(isom2, jsom2).H(isom1, jsom1);
                                end
                            end
                        end
                    end
                    % find the cross-modal best-matching-unit (max activity)
                    [bmval, bmloc] = max(cross_mod2(:)); [xid, yid] = ind2sub(size(cross_mod2), bmloc);
                    bmuind1.xpos = xid; bmuind1.ypos = yid; bmuind1.act = bmval;
                    som1(idx, jdx).ai = exp(-(norm([bmuind1.xpos - idx, bmuind1.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)));
                    
                    % from all other units in SOM1
                    for isom2 = 1:simopts.net.size
                        for jsom2 = 1:simopts.net.size
                            for isom1 = 1:simopts.net.size
                                for jsom1 = 1:simopts.net.size
                                    % sum of all products between direct activation
                                    % and cross-som Hebbian weights
                                    cross_mod1(isom2, jsom2) = cross_mod1(isom2, jsom2) + ...
                                        som1(isom1, jsom1).ad*som1(isom1, jsom1).H(isom2, jsom2);
                                end
                            end
                        end
                    end
                    % find the cross-modal best-matching-unit
                    [bmval, bmloc] = max(cross_mod1(:)); [xid, yid] = ind2sub(size(cross_mod1), bmloc);
                    bmuind2.xpos = xid; bmuind2.ypos = yid; bmuind2.act = bmval;
                    som2(idx, jdx).ai = exp(-(norm([bmuind2.xpos - idx, bmuind2.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)));
                    
                    % ---------------------------------------------------------------------------------------------------------
                    
                    % compute the joint activation from both afferent
                    % sensory input and cross-modal Hebbian linkage for
                    
                    % SOM1
                    som1(idx, jdx).at = (1 - gammat(net_iter))*som1(idx, jdx).ad + gammat(net_iter)*som1(idx, jdx).ai;
                    
                    % SOM2
                    som2(idx, jdx).at = (1 - gammat(net_iter))*som2(idx, jdx).ad + gammat(net_iter)*som2(idx, jdx).ai;
                    
                end
            end
            
            % check if verbose is on for debugging
            if(simopts.debug.verbose == 1)
                fprintf('-----------Indirect Activation Phase-----------\n');
                % first som
                fprintf(1, 'SOM1BMU_IND = (%d,%d) with ad = %f \n', bmuind1.xpos, bmuind1.ypos, som1(bmudir1.xpos, bmudir1.ypos).ad);
                fprintf(1, 'SOM1.ai = \n'); [som1.ai]
                % second som
                fprintf(1, 'SOM2BMU_IND = (%d,%d) with ad = %f \n', bmuind2.xpos, bmuind2.ypos, som2(bmudir2.xpos, bmudir2.ypos).ad);
                fprintf(1, 'SOM2.ai = \n'); [som2.ai]
                
                fprintf('-------------Total Activation Phase------------\n');
                % first som
                fprintf(1, 'SOM1.at = \n'); [som1.at]
                % second som
                fprintf(1, 'SOM2.at = \n'); [som2.ad]
            end
            
            % check if debug visualization is on
            if(simopts.debug.visual == 1)
                % epoch wise visualization of network activities(direct, indirect and total)
                figure(act_vis);
                % --------------------------------------------------------------------------------------
                % time delay for visualization
                vis_time_delay = 0; % s
                % direct activities for the 2 som
                % som1
                subplot(3,2,1);
                for idx = 1:simopts.net.size
                    for jdx = 1:simopts.net.size
                        visual_somd1(idx, jdx) = som1(idx, jdx).ad;
                    end
                end
                imagesc(visual_somd1(1:simopts.net.size, 1:simopts.net.size)); colormap; colorbar; axis xy; pause(vis_time_delay);
                title('SOM1 Direct activation');
                % som2
                subplot(3,2,2);
                for idx = 1:simopts.net.size
                    for jdx = 1:simopts.net.size
                        visual_somd2(idx, jdx) = som2(idx, jdx).ad;
                    end
                end
                imagesc(visual_somd2(1:simopts.net.size, 1:simopts.net.size)); colormap; colorbar;  axis xy; pause(vis_time_delay);
                title('SOM2 Direct activation');
                % --------------------------------------------------------------------------------------
                % indirect activities for the 2 som
                subplot(3,2,3);
                % som1
                for idx = 1:simopts.net.size
                    for jdx = 1:simopts.net.size
                        visual_somi1(idx, jdx) = som1(idx, jdx).ai;
                    end
                end
                imagesc(visual_somi1(1:simopts.net.size, 1:simopts.net.size)); colormap; colorbar;  axis xy; pause(vis_time_delay);
                title('SOM1 Indirect activation');
                subplot(3,2,4);
                % som2
                for idx = 1:simopts.net.size
                    for jdx = 1:simopts.net.size
                        visual_somi2(idx, jdx) = som2(idx, jdx).ai;
                    end
                end
                imagesc(visual_somi2(1:simopts.net.size, 1:simopts.net.size)); colormap; colorbar;  axis xy; pause(vis_time_delay);
                title('SOM2 Indirect activation');
                % --------------------------------------------------------------------------------------
                % total activities for the 2 som
                subplot(3,2,5);
                % som1
                for idx = 1:simopts.net.size
                    for jdx = 1:simopts.net.size
                        visual_somt1(idx, jdx) = som1(idx, jdx).at;
                    end
                end
                imagesc(visual_somt1(1:simopts.net.size, 1:simopts.net.size)); colormap; colorbar;  axis xy; pause(vis_time_delay);
                title('SOM1 Total activation');
                subplot(3,2,6);
                % som2
                for idx = 1:simopts.net.size
                    for jdx = 1:simopts.net.size
                        visual_somt2(idx, jdx) = som2(idx, jdx).at;
                    end
                end
                imagesc(visual_somt2(1:simopts.net.size, 1:simopts.net.size)); colormap; colorbar;  axis xy; pause(vis_time_delay);
                title('SOM2 Total activation'); pause(vis_time_delay);
                suptit = sprintf('Activities in the 2 som networks - %d Epochs', net_iter);
                suptitle(suptit);
                % --------------------------------------------------------------------------------------
            end % env verbose and visualization
            
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                    % ---------------------------------------------------------------------------------------------------------
                    
                    % update weights for the current neuron in the BMU
                    % neighborhood - move the neurons weights close to the input
                    % pattern according to the total activation pattern
                    
                    % input weights update combining an excitatory and
                    % inhibitory component such that a unit is brought
                    % closer to the input if is activated by BOTH afferent senory input and
                    % cross modal projections (co-activation pattern)
                    
                    % update SOM1
                    for w_idx = 1:simopts.data.trainvsize
                        som1(idx, jdx).W(w_idx)= som1(idx, jdx).W(w_idx) + alphat(net_iter)*som1(idx, jdx).at*(netin.trainv1(trainv_idx, w_idx) - som1(idx, jdx).W(w_idx))-...
                            xit(net_iter)*(som1(idx, jdx).ad - som1(idx, jdx).at)*(netin.trainv1(trainv_idx, w_idx) - som1(idx, jdx).W(w_idx));
                    end
                    
                    % check if verbose is on for debugging
                    if(simopts.debug.verbose == 1)
                        fprintf('-----------Map Update Phase------------\n');
                        % first som
                        fprintf(1, 'SOM1_RAW(%d,%d).W = \n', idx, jdx); som1(idx, jdx).W
                        fprintf(1, 'SOM1.W = \n'); som1.W
                        
                    end
                    
                    % update SOM2
                    for w_idx = 1:simopts.data.trainvsize
                        som2(idx, jdx).W(w_idx) = som2(idx, jdx).W(w_idx) + alphat(net_iter)*som2(idx, jdx).at*(netin.trainv2(trainv_idx, w_idx) - som2(idx, jdx).W(w_idx))-...
                            xit(net_iter)*(som2(idx, jdx).ad - som2(idx, jdx).at)*(netin.trainv2(trainv_idx, w_idx) - som2(idx, jdx).W(w_idx));
                    end
                    
                    % check if verbose is on for debugging
                    if(simopts.debug.verbose == 1)
                        % second som
                        fprintf(1, 'SOM2_RAW(%d,%d).W = \n', idx, jdx); som2(idx, jdx).W
                        fprintf(1, 'SOM2.W = \n'); som2.W
                    end
                    
                end
            end
            
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                    % ---------------------------------------------------------------------------------------------------------
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                    
                    % update for SOM1
                    for isom2 = 1:simopts.net.size
                        for jsom2 = 1:simopts.net.size
                            switch simopts.net.xmodlearn
                                case 'hebb'
                                    % compute new weight using Hebbian rule
                                    % deltaH = K*preH*postH
                                    som1(idx, jdx).H(isom2, jsom2)= som1(idx, jdx).H(isom2, jsom2) + kappat(net_iter)*som1(idx, jdx).at*som2(isom2, jsom2).at;
                                case 'covariance'
                                    % compute new weight using covariance learning (pseudo-Hebbian) rule
                                    % deltaH = K*(preH - mean(preH))*
                                    %            (postH - mean(postH));
                                    som1(idx, jdx).H(isom2, jsom2)= som1(idx, jdx).H(isom2, jsom2) + kappat(net_iter)*(som1(idx, jdx).at - mean([som1.at]))*(som2(isom2, jsom2).at - mean([som2.at]));
                            end
                        end
                    end
                    
                    % check if verbose is on for debugging
                    if(simopts.debug.verbose == 1)
                        % first som
                        fprintf(1, 'SOM1_RAW(%d,%d).H= \n', idx, jdx); som1(idx, jdx).H
                    end
                    
                    % update for SOM2
                    for isom1 = 1:simopts.net.size
                        for jsom1 = 1:simopts.net.size
                            
                            switch simopts.net.xmodlearn
                                case 'hebb'
                                    % compute new weight using Hebbian rule
                                    % deltaH = K*preH*postH
                                    som2(idx, jdx).H(isom1, jsom1)= som2(idx, jdx).H(isom1, jsom1) + kappat(net_iter)*som2(idx, jdx).at*som1(isom1, jsom1).at;
                                case 'covariance'
                                    % compute new weight using covariance learning (pseudo-Hebbian) rule
                                    % deltaH = K*(preH - mean(preH))*
                                    %            (postH - mean(postH));
                                    som2(idx, jdx).H(isom1, jsom1)= som2(idx, jdx).H(isom1, jsom1) + kappat(net_iter)*(som2(idx, jdx).at - mean([som2.at]))*(som1(isom1, jsom1).at - mean([som1.at]));
                            end
                        end
                    end
                    
                    % check if verbose is on for debugging
                    if(simopts.debug.verbose == 1)
                        % second som
                        fprintf(1, 'SOM2_RAW(%d,%d).H= \n', idx, jdx); som2(idx, jdx).H
                    end
                end
            end
            
            % cross-modal Hebbian links normalization and update
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                    
                    for isom2 = 1:simopts.net.size
                        for jsom2 = 1:simopts.net.size
                            % normalize weights
                            som1(idx, jdx).H(isom2, jsom2) = (som1(idx, jdx).H(isom2, jsom2) - min(min([som1.H])))/(max(max([som1.H])) - min(min([som1.H])));
                        end
                    end
                    
                    % check if verbose is on for debugging
                    if(simopts.debug.verbose == 1)
                        % first som
                        fprintf(1, 'SOM1_NORM(%d,%d).H= \n', idx, jdx); som1(idx, jdx).H
                        fprintf(1, 'SOM1.H = \n'); som1.H
                    end
                    
                    for isom1 = 1:simopts.net.size
                        for jsom1 = 1:simopts.net.size
                            % normalize weights
                            som2(idx, jdx).H(isom1, jsom1) = (som2(idx, jdx).H(isom1, jsom1) - min(min([som2.H])))/(max(max([som2.H])) - min(min([som2.H])));
                        end
                    end
                    
                    % check if verbose is on for debugging
                    if(simopts.debug.verbose == 1)
                        % second som
                        fprintf(1, 'SOM2_NORM(%d,%d).H= \n', idx, jdx); som2(idx, jdx).H
                        fprintf(1, 'SOM2.H = \n'); som2.H
                    end
                end
            end
            
            % check if debug visualization is on
            if(simopts.debug.visual == 1)
                % epoch wise visualization of hebbian cross-modal
                % weight matrices for som1
                figure(hweight_vis1);
                % --------------------------------------------------------------------------------------
                for sidx = 1:rown*coln
                    subplot(rown, coln, sidx);
                    [ridx, cidx] = ind2sub([coln, rown], sidx);
                    % plot the weights for current neuron
                    imagesc(som1(cidx, ridx).H(1:simopts.net.size, 1:simopts.net.size)); hold on;
                    %if(ridx == simopts.net.size)
                    colorbar;
                    %end
                    axis xy; colormap; box off;
                end
                suptit1 = sprintf('Cross-modal weights of SOM1 - %d Epochs', net_iter);
                suptitle(suptit1);
                % weight matrices for som2
                figure(hweight_vis2);
                % --------------------------------------------------------------------------------------
                for sidx = 1:rown*coln
                    subplot(rown, coln, sidx);
                    [ridx, cidx] = ind2sub([coln, rown], sidx);
                    % plot the weights for current neuron
                    imagesc(som2(cidx, ridx).H(1:simopts.net.size, 1:simopts.net.size)); hold on;
                    %if(ridx == simopts.net.size)
                    colorbar;
                    %end
                    axis xy; colormap; box off;
                end
                suptit2 = sprintf('Cross-modal weights of SOM2 - %d Epochs', net_iter);
                suptitle(suptit2);
                % --------------------------------------------------------------------------------------
            end % env debug visualization
        end % end for each entry in the training vector
        net_iter = net_iter + 1;
    else
        disp 'cln_iterate_network: Finalized training phase.'
        break;
    end
end
% save everything to a file and return the name
file_dump = sprintf('%d_epochs_%d_neurons_%s_source_data_%s_correlation_%d_trainvsize_%d_trainvnum_%d_params',...
            simopts.net.maxepochs, simopts.net.size, simopts.data.source, simopts.data.corrtype, simopts.data.trainvsize,...
            simopts.data.ntrainv, simopts.net.params);
save(file_dump);
rundata = load(file_dump);
end
