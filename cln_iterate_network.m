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
qedir1 = zeros(simopts.net.size, simopts.net.size);
qeind1 = zeros(simopts.net.size, simopts.net.size);
qedir2 = zeros(simopts.net.size, simopts.net.size);
qeind2 = zeros(simopts.net.size, simopts.net.size);
sum_norm_W1 = zeros(1, simopts.data.trainvsize); sum_norm_W2 = zeros(1, simopts.data.trainvsize);
cross_mod1 = 0; cross_mod2 = 0;
% init learning rate and neighborhood radius
alphat = zeros(1, simopts.net.maxepochs);
sigmat = zeros(1, simopts.net.maxepochs);
tau = 1000;
% main loop of the network
while(1)
    % check if we finished training
    if(net_iter <= simopts.net.maxepochs)
        % present a vector from each training data set to the network's
        % SOMs
        for trainv_idx = 1:netin.trainsetsize
            % max quantization error init
            qe_max1_dir = Inf; qe_max2_dir = Inf;
            qe_max1_ind = Inf; qe_max2_ind = Inf;
            % go through the two SOMs
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                    % compute quantization errors in each map for
                    % direct projections from the sensors
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
                    % compute quantization errors in each map for
                    % indirect cross-modal projections
                    for som2idx = 1:simopts.net.size
                        for som2jdx = 1:simopts.net.size
                            qeind1(idx, jdx) = norm(som2(som2idx, som2jdx).H - som1(idx, jdx).H);
                            % check if current neuron is winner
                            if(qeind1(idx, jdx)<qe_max1_ind)
                                bmuind1.xpos = idx;
                                bmuind1.ypos = jdx;
                                bmuind1.qe = qeind1(idx, jdx);
                                qe_max1_ind = qeind1(idx, jdx);
                            end
                        end
                    end
                    for som1idx = 1:simopts.net.size
                        for som1jdx = 1:simopts.net.size
                            qeind2(idx, jdx) = norm(som1(som1idx, som1jdx).H - som2(idx, jdx).H);
                            % check if current neuron is winner
                            if(qeind2(idx, jdx)<qe_max2_ind)
                                bmuind2.xpos = idx;
                                bmuind2.ypos = jdx;
                                bmuind2.qe = qeind2(idx, jdx);
                                qe_max2_ind = qeind2(idx, jdx);
                            end

                        end
                    end
                end
            end % end of bmus search loop
          
            % compute the activations of all nodes in the BMU neighborhood
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                   %-------------------------------------------------------------------------------
                    % use the same leraning parameters for both SOM
                    % compute the learning rate @ current epoch
                    
                    % exponential learning rate adaptation
                    alphat(net_iter) = simopts.net.alpha*exp(-net_iter/tau);
                    
                    % semi-empirical learning rate adaptation
%                     A = simopts.net.maxepochs/100.0; B = A;
%                     alphat(net_iter) = A/(net_iter + B);
%                     
                    % compute the neighborhood radius size @ current epoch
                    sigmat(net_iter) = simopts.net.sigma*exp(-net_iter/simopts.net.lambda);
                    %-------------------------------------------------------------------------------
                    % fist SOM activations
                    % compute the direct activation - neighborhood kernel
                    som1(idx, jdx).ad = alphat(net_iter)*...
                        exp(-(norm([bmudir1.xpos - idx, bmudir1.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)))*(1 - qedir1(idx, jdx));
                    
                    % compute the indirect activation (from all other units in SOM2)
                    % first compute the total activation from the other SOM
                    % projected via the Hebbian links
                    for isom2 = 1:simopts.net.size
                        for jsom2 = 1:simopts.net.size
                            % sum of all products between direct activation
                            % and cross-som Hebbian weights
                            cross_mod2 = cross_mod2 + ...
                                som2(isom2, jsom2).ad*som1(idx, jdx).H(isom2, jsom2);
                        end
                    end
                    som1(idx, jdx).ai = cross_mod2;
                    
                    % compute the joint activation from both input space
                    % and cross-modal Hebbian linkage
                    
                    som1(idx, jdx).at = (1 - simopts.net.gamma)*exp(-(norm([bmudir1.xpos - idx, bmudir1.ypos - jdx]))^2/(2*(sigmat(net_iter)^2))) + ...
                        simopts.net.gamma*exp(-(norm([bmuind1.xpos - idx,  bmuind1.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)));
                    
                    % update weights for the current neuron in the BMU
                    
                    % normalize weights from input space
                    % compute the sum squared weight update for normalization
                    for w_idx = 1:simopts.data.trainvsize
                        for norm_idx = 1:simopts.net.size
                            for norm_jdx = 1:simopts.net.size
                                sum_norm_W1(w_idx) = sum_norm_W1(w_idx) + (som1(norm_idx, norm_jdx).W(w_idx) + alphat(net_iter)*som1(norm_idx, norm_jdx).at*qedir1(norm_idx, norm_jdx)-...
                                    simopts.net.xi*(som1(norm_idx, norm_jdx).ad - som1(norm_idx, norm_jdx).at)*qedir1(norm_idx, norm_jdx))^2;
                            end
                        end
                    end
                    
                    % input weights update combining an excitatory and
                    % inhibitory component such that a unit is brought
                    % closer to the input if is activated by BOTH input and
                    % cross modal input
                    for w_idx = 1:simopts.data.trainvsize
                        som1(idx, jdx).W(w_idx) = (som1(idx, jdx).W(w_idx) + alphat(net_iter)*som1(idx, jdx).at*qedir1(idx, jdx)-...
                            simopts.net.xi*(som1(idx, jdx).ad - som1(idx, jdx).at)*(netin.trainv1(trainv_idx, w_idx) - som1(idx, jdx).W(w_idx)))/...
                            sqrt(sum_norm_W1(w_idx));
                    end
                    
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                    for isom2 = 1:simopts.net.size
                        for jsom2 = 1:simopts.net.size
                            
                            % compute new weight using Hebbian learning
                            % rule deltaH = K*preH*postH
                            som1(idx, jdx).H(isom2, jsom2)= som1(idx, jdx).H(isom2, jsom2)+ simopts.net.kappa*som1(idx, jdx).at*som2(idx, jdx).at;
                        end
                    end
                    
                    %-------------------------------------------------------------------------------
                    % compute second SOM activations
                    
                    % compute the direct activation - neighborhood kernel
                    som2(idx, jdx).ad = alphat(net_iter)*...
                        exp(-(norm([bmudir2.xpos - idx, bmudir2.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)))*(1-qedir2(idx, jdx));
                    
                    % compute the indirect activation (from all other units in SOM2)
                    % first compute the total activation from the other SOM
                    % via the Hebbian links
                    for isom1 = 1:simopts.net.size
                        for jsom1 = 1:simopts.net.size
                            % sum of all products between direct activation
                            % and cross-som Hebbian weights
                            cross_mod1 = cross_mod1 + ...
                                som1(isom1, jsom1).ad*som2(idx, jdx).H(isom1, jsom1);
                        end
                    end
                    som2(idx, jdx).ai = cross_mod1;
                    
                    % compute the joint activation from both input space
                    % and cross-modal Hebbian linkage
                    
                    som2(idx, jdx).at = (1 - simopts.net.gamma)*exp(-(norm([idx - bmudir2.xpos, jdx - bmudir2.ypos]))^2/(2*(sigmat(net_iter)^2))) + ...
                        simopts.net.gamma*exp(-(norm([idx - bmuind2.xpos, jdx - bmuind2.ypos]))^2/(2*(sigmat(net_iter)^2)));
                    
                    % update weights for the current neuron in the BMU
                    
                    % normalize weights from input space
                    % compute the sum squared weight update for normalization
                    for w_idx = 1:simopts.data.trainvsize
                        for norm_idx = 1:simopts.net.size
                            for norm_jdx = 1:simopts.net.size
                                sum_norm_W2(w_idx) = sum_norm_W2(w_idx) + (som2(norm_idx, norm_jdx).W(w_idx) + alphat(net_iter)*som2(norm_idx, norm_jdx).at*qedir2(norm_idx, norm_jdx)-...
                                    simopts.net.xi*(som2(norm_idx, norm_jdx).ad - som2(norm_idx, norm_jdx).at)*qedir2(norm_idx, norm_jdx))^2;
                            end
                        end
                    end
                    
                    % input weights update combining an excitatory and
                    % inhibitory component such that a unit is brought
                    % closer to the input if is activated by BOTH input and
                    % cross modal input
                    for w_idx = 1:simopts.data.trainvsize
                        som2(idx, jdx).W(w_idx) = (som2(idx, jdx).W(w_idx) + alphat(net_iter)*som2(idx, jdx).at*qedir2(idx, jdx)-...
                            simopts.net.xi*(som2(idx, jdx).ad - som2(idx, jdx).at)*(netin.trainv2(trainv_idx, w_idx) - som2(idx, jdx).W(w_idx)))/...
                            sqrt(sum_norm_W2(w_idx));
                        
                    end
                    
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                    for isom1 = 1:simopts.net.size
                        for jsom1 = 1:simopts.net.size
                            
                            % update weight using Hebbian learning rule
                            som2(idx, jdx).H(isom1, jsom1)= som2(idx, jdx).H(isom1, jsom1) + simopts.net.kappa*som2(idx, jdx).at*som1(idx, jdx).at;
                        end
                    end
                    
                    %-------------------------------------------------------------------------------
                    
                end
            end           
        end % end for each entry in the training vector
        net_iter = net_iter + 1;
    else
        disp 'cln_iterate_network: Finalized training phase.'
        break;
    end
end
    % save everything to a file and return the name
    file_dump = sprintf('%d_epochs_%s', simopts.net.maxepochs, simopts.data.corrtype);
    save(file_dump);
    rundata = load(file_dump);
end
