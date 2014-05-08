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
%   vishdl - figures handles for later access

function somfig = cln_visualize_som(visin, som)
somfig = figure;
set(gcf, 'color', 'white'); box off; grid off;
% display template
COLS = 6; ROWS = 6;
% plot the learning learning adaptation
subplot(ROWS, COLS, [5,12]); plot(visin.alphat, '.b');
% plot the neighborhood kernel radius adaptation
hold on; plot(visin.sigmat, '.r'); 
hold on; plot(visin.gammat, '.g');
hold on; plot(visin.xit, '*m');
hold on; plot(visin.kappat, '*k');
title('Adaptation parameters'); xlabel('Epochs');
legend('Learning rate','Neighborhood kernel radius','Total activation gain param','Inhibitory gain in W update','Hebbian learning rate in cross-modal interaction'); box off;
% total activity in each neurons in the SOM
subplot(ROWS, COLS, [3,10]);
at_vis = zeros(visin.simopts.net.size, visin.simopts.net.size);
for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        at_vis(idx, jdx) = som(idx, jdx).at;
    end
end
imagesc((at_vis(1:visin.simopts.net.size, 1:visin.simopts.net.size))); colorbar; axis xy;
colormap; box off; title('Total (joint) activity in the network');

% total activity in each neurons in the SOM - alternative 3D surfed display
subplot(ROWS, COLS, [1,8]);
surf(1:visin.simopts.net.size, 1:visin.simopts.net.size, at_vis(1:visin.simopts.net.size, 1:visin.simopts.net.size));
axis xy; colorbar;box off; title('Total (joint) activity in the network');

% direct activity elicited by sensory projections (plastic connections)
subplot(ROWS, COLS, [13, 20]);
ad_vis = zeros(visin.simopts.net.size, visin.simopts.net.size);
for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        ad_vis(idx, jdx) = som(idx, jdx).ad;
    end
end
imagesc((ad_vis(1:visin.simopts.net.size, 1:visin.simopts.net.size))); set(gcf, 'color', 'white'); colorbar; axis xy;
colormap; box off; title('Sensory elicited act.');
% indirect activity elicited by cross-modal Hebbian linkage (plastic connections)
subplot(ROWS, COLS, [17, 24]);
ai_vis = zeros(visin.simopts.net.size, visin.simopts.net.size);
for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        ai_vis(idx, jdx) = som(idx, jdx).ai;
    end
end
imagesc((ai_vis(1:visin.simopts.net.size, 1:visin.simopts.net.size))); set(gcf, 'color', 'white');colorbar; axis xy;
colormap; box off; title('Cross-modal elicited act.');
% synaptic connections strenghts from sensory projections (W weight matrix)
subplot(ROWS, COLS, [25, 32]);
W_vis_elem = zeros(visin.simopts.net.size, visin.simopts.net.size, visin.simopts.net.size, visin.simopts.net.size);
for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        for elem_idx = 1:visin.simopts.net.size
            W_vis_elem(idx, jdx, elem_idx) = som(idx, jdx).W(elem_idx);
            scatter3(idx, jdx, elem_idx, 30, W_vis_elem(idx, jdx, elem_idx), 'filled');
            hold on; colorbar; set(gcf, 'color', 'white');
        end
    end
end
box off; title('Sensory projections synaptic weights'); axis xy; axis equal;

% synaptic connections strenghts from cross modal Hebbian interaction (H weight matrix)
subplot(ROWS, COLS, [27, 34]);

H_vis_elem = zeros(visin.simopts.net.size, visin.simopts.net.size, visin.simopts.net.size, visin.simopts.net.size);

for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        for idx_elem = 1:visin.simopts.net.size
            for jdx_elem = 1:visin.simopts.net.size
                H_vis_elem(idx, jdx, idx_elem, jdx_elem) = som(idx, jdx).H(idx_elem, jdx_elem);
            end
        end
    end
end

% group everything in a horizontal matrix
temp_horz = zeros(visin.simopts.net.size, visin.simopts.net.size);
temp_proj_mat = zeros(visin.simopts.net.size, visin.simopts.net.size);
for iidx = 1:visin.simopts.net.size
    for ijdx = 1:visin.simopts.net.size
        % extract the (iidx, ijdx) th entry in the multidim matrix
        temp_lin_submat = H_vis_elem(iidx, ijdx, :);
        % remove singleton dimensions
        temp_lin_squeezed = squeeze(temp_lin_submat);
        % restore indices and value in visin.simopts.net.sizexvisin.simopts.net.size matrix
        for idx_temp = 1:length(temp_lin_squeezed)
            [jdx_proj, idx_proj] = ind2sub([visin.simopts.net.size, visin.simopts.net.size], idx_temp);
            temp_proj_mat(idx_proj, jdx_proj) = temp_lin_squeezed(idx_temp);
        end
        temp_horz =  horzcat(temp_horz, temp_proj_mat);
    end
end

% flatten down the horizontal buffer to a square matrix for visualization
collapsed_view = zeros(visin.simopts.net.size*visin.simopts.net.size, visin.simopts.net.size*visin.simopts.net.size);
% linear index of the temp horizontal buffer temp_idx

for temp_idx = 1:visin.simopts.net.size*visin.simopts.net.size*visin.simopts.net.size*visin.simopts.net.size
    [conv_idx, conv_jdx] = ind2sub(size(collapsed_view), temp_idx);
    collapsed_view(conv_idx, conv_jdx) = temp_horz(temp_idx);
end

imagesc(collapsed_view(1:visin.simopts.net.size*visin.simopts.net.size, 1:visin.simopts.net.size*visin.simopts.net.size)); hold on; colorbar; axis xy;
colormap; box off; title('Cross-modal synaptic weights');


% synaptic connections strenghts from cross modal Hebbian interaction (H
% weight matrix) - alternative display using overlapping surfs
subplot(ROWS, COLS, [29, 36]);

H_vis_elem = zeros(visin.simopts.net.size, visin.simopts.net.size, visin.simopts.net.size, visin.simopts.net.size);

for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
        for idx_elem = 1:visin.simopts.net.size
            for jdx_elem = 1:visin.simopts.net.size
                H_vis_elem(idx, jdx, idx_elem, jdx_elem) = som(idx, jdx).H(idx_elem, jdx_elem);
            end
        end
    end
end

% plot overlappig surfs
for idx = 1:visin.simopts.net.size
    for jdx = 1:visin.simopts.net.size
                surf(som(idx, jdx).H(1:visin.simopts.net.size, 1:visin.simopts.net.size)); hold on; axis xy; colorbar;
    end
end
box off; title('Cross-modal synaptic weights');

end

