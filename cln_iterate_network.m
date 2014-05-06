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
net_iter = 2;
% quantization error for each map and intermap
qedir1 = zeros(simopts.net.size, simopts.net.size);
qedir2 = zeros(simopts.net.size, simopts.net.size);
cross_mod1 = zeros(simopts.net.size, simopts.net.size); 
cross_mod2 = zeros(simopts.net.size, simopts.net.size);
% init adaptive params
alphat = zeros(1, simopts.net.maxepochs);
sigmat = zeros(1, simopts.net.maxepochs);
gammat = zeros(1, simopts.net.maxepochs);
xit = zeros(1, simopts.net.maxepochs);
kappat = zeros(1, simopts.net.maxepochs);
% set up the initial values 
alphat(1) = simopts.net.alpha;
sigmat(1) = simopts.net.sigma;
gammat(1) = simopts.net.gamma;
xit(1)    = simopts.net.xi;
kappat(1) = simopts.net.kappa;
tau = 1000;
% main loop of the network
while(1)
    % check if we finished training
    if(net_iter <= simopts.net.maxepochs)
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
        
            % compute the activations for each neuron in each som s
            for idx = 1:simopts.net.size
                for jdx = 1:simopts.net.size
                   %-------------------------------------------------------------------------------
                    % use the same learning parameters for both SOM
                    
                    % for thresholding of the adaptive parameters use 
                    %   param_out = min( max( param_in, min_value ), max_value ); 
                    
                    % the learning rate and radius decrease over time to
                    % enable learning on a coarse and then on a finer time
                    % scale
                    
                    % compute the learning rate @ current epoch
                    
                    % exponential learning rate adaptation
                    % alphat(net_iter) = simopts.net.alpha*exp(-net_iter/tau);
                    alphat(net_iter) = alphat(net_iter-1) * 0.99;
                    
                    % semi-empirical learning rate adaptation
                    % A = simopts.net.maxepochs/100.0; B = A;
                    % alphat(net_iter) = A/(net_iter + B);  
                   
                    % compute the neighborhood radius size @ current epoch
                    % sigmat(net_iter) = simopts.net.sigma*exp(-net_iter/simopts.net.lambda);
                    sigmat(net_iter) = sigmat(net_iter-1)^(-net_iter/tau);                  
                    
                    % adapt the cross-modal interaction params (increase in time)
                    
                    % cross-modal activation impact on local som learning
                    % gammat(net_iter) = simopts.net.gamma*exp(net_iter/tau);
                    gammat(net_iter) = gammat(net_iter-1)*1.01;
                    if(gammat(net_iter)>1) 
                        gammat(net_iter) = 1; 
                    end
                    
                    % inhibitory component to ensure only co-activation
                    % xit(net_iter) = simopts.net.xi*exp(net_iter/tau);
                    xit(net_iter) = xit(net_iter - 1)*1.015;
                    if(xit(net_iter)>0.07)
                        xit(net_iter) = 0.07;
                    end
                                                        
                    % Hebbian learning rate
                    % kappat(net_iter) = simopts.net.kappa*exp(net_iter/tau);
                    kappat(net_iter) = kappat(net_iter)*1.01;
                    if(kappat(net_iter)>0.3)
                        kappat(net_iter) = 0.3;
                    end
    
                    %-------------------------------------------------------------------------------
                    % fist SOM activations
                    
                    % compute the direct activation - neighborhood kernel
                    som1(idx, jdx).ad = exp(-(norm([bmudir1.xpos - idx, bmudir1.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)));
                    
                    % compute the indirect activation (from all other units in SOM2)
                    for isom1 = 1:simopts.net.size
                        for jsom1 = 1:simopts.net.size
                            for isom2 = 1:simopts.net.size
                                for jsom2 = 1:simopts.net.size
                                    % sum of all products between direct activation
                                    % and cross-som Hebbian weights
                                    cross_mod2(isom1, jsom1) = cross_mod2(isom1, jsom1) + ...
                                        som2(isom2, jsom2).ad*som1(isom2, jsom2).H(isom1, jsom1);
                                end
                            end
                        end
                    end
                    % find the cross-modal best-matching-unit (max activity)
                    [bmval, bmloc] = max(cross_mod2(:)); [xid, yid] = ind2sub(size(cross_mod2), bmloc);
                    bmuind1.xpos = xid; bmuind1.ypos = yid; bmuind1.act = bmval;
                    som1(idx, jdx).ai = exp(-(norm([bmuind1.xpos - idx, bmuind1.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)));
                                                        
                    
                    % compute the joint activation from both afferent
                    % sensory input and cross-modal Hebbian linkage
                    som1(idx, jdx).at = (1 - gammat(net_iter))*som1(idx, jdx).ad + gammat(net_iter)*som1(idx, jdx).ai;
                    
                    % update weights for the current neuron in the BMU
                    % neighborhood - move the neurons weights close to the input
                    % pattern according to the total activation pattern
                    
                    % input weights update combining an excitatory and
                    % inhibitory component such that a unit is brought
                    % closer to the input if is activated by BOTH afferent senory input and
                    % cross modal projections (co-activation pattern)
                    for w_idx = 1:simopts.data.trainvsize
                        som1(idx, jdx).W(w_idx) = som1(idx, jdx).W(w_idx) + alphat(net_iter)*som1(idx, jdx).at*qedir1(idx, jdx)-...
                            xit(net_iter)*(som1(idx, jdx).ad - som1(idx, jdx).at)*qedir1(idx, jdx);
                        % normallize weights
                        som1(idx, jdx).W(w_idx) = (som1(idx, jdx).W(w_idx) - min((som1(idx, jdx).W(:))))/max((som1(idx, jdx).W(:)));
                    end
                    
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                                                            
                    for isom2 = 1:simopts.net.size
                        for jsom2 = 1:simopts.net.size
                            % compute new weight using Hebbian rule
                            % rule deltaH = K*preH*postH
                            som1(idx, jdx).H(isom2, jsom2)= som1(idx, jdx).H(isom2, jsom2) + kappat(net_iter)*som1(idx, jdx).at*som2(isom2, jsom2).at;
                            % normalize weights
                            som1(idx, jdx).H(isom2, jsom2) = (som1(idx, jdx).H(isom2, jsom2) - min((som1(idx, jdx).H(:))))/max((som1(idx, jdx).H(:))) ;
                        end
                    end
                                       
                    %-------------------------------------------------------------------------------
                    % compute second SOM activations
                    
                    % compute the direct activation - neighborhood kernel
                    som2(idx, jdx).ad = exp(-(norm([bmudir2.xpos - idx, bmudir2.ypos - jdx]))^2/(2*(sigmat(net_iter)^2)));
                    
                    % compute the indirect activation (from all other units in SOM1)
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
                                                        
                    
                    % compute the joint activation from both afferent
                    % sensory input and cross-modal Hebbian linkage
                    som2(idx, jdx).at = (1 - gammat(net_iter))*som2(idx, jdx).ad + gammat(net_iter)*som2(idx, jdx).ai;
                    
                    % update weights for the current neuron in the BMU
                    % neighborhood - move the neurons close to the input
                    % pattern according to the total activation pattern
                    
                    % input weights update combining an excitatory and
                    % inhibitory component such that a unit is brought
                    % closer to the input if is activated by BOTH afferent senory input and
                    % cross modal projections (co-activation pattern)
                    for w_idx = 1:simopts.data.trainvsize
                        som2(idx, jdx).W(w_idx) = som2(idx, jdx).W(w_idx) + alphat(net_iter)*som2(idx, jdx).at*qedir2(idx, jdx)-...
                            xit(net_iter)*(som2(idx, jdx).ad - som2(idx, jdx).at)*qedir2(idx, jdx);
                        % normallize weights
                        som2(idx, jdx).W(w_idx) = (som2(idx, jdx).W(w_idx) - min((som2(idx, jdx).W(:))))/max((som2(idx, jdx).W(:)));
                    end
                    
                    % cross-modal Hebbian links update for co-activated
                    % neurons in both SOMs
                                                            
                    for isom1 = 1:simopts.net.size
                        for jsom1 = 1:simopts.net.size
                            % compute new weight using Hebbian rule
                            % rule deltaH = K*preH*postH
                            som2(idx, jdx).H(isom1, jsom1)= som2(idx, jdx).H(isom1, jsom1) + kappat(net_iter)*som2(idx, jdx).at*som1(isom1, jsom1).at;
                            % normalize weights
                            som2(idx, jdx).H(isom1, jsom1) = (som2(idx, jdx).H(isom1, jsom1) - min((som2(idx, jdx).H(:))))/max((som2(idx, jdx).H(:))) ;
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
