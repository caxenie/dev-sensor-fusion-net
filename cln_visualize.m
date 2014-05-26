%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% For the current implementation we only consider a 2 variable network

% FUNCTION
% Performs the visualization of the network data.
%
% ARGS
%   visin - network data to visualize
%
% RETURN
%   vishdl - figures handles for later access

function vishdl = cln_visualize(visin)
% visualize input data
vishdl(1) = figure; 
set(gcf, 'color', 'white');
subplot(2, 1, 1);plot(visin.netin.raw1, '.b'); box off; grid off;
title('Input var P1'); xlabel('Samples');
subplot(2, 1, 2);plot(visin.netin.raw2, '.g'); box off; grid off;
title('Input var P2'); xlabel('Samples');
vishdl(3) = figure;
set(gcf, 'color', 'white'); box off; grid off;
plot(visin.alphat, '.b');
% plot the neighborhood kernel radius adaptation
hold on; plot(visin.gammat, '.g');
hold on; plot(visin.xit, '.m'); hold on; plot(visin.kappat, '.k');
suptitle('Adaptation parameters'); xlabel('Epochs');
legend('Learning rate','Cross-modal impact','Inhibitory gain in W update','Hebbian learning rate in cross-modal interaction'); box off;
vishdl(4) = figure;
set(gcf, 'color', 'white');
hold on; plot(visin.sigmat, '.r');legend('Neighborhood radius adaptation');xlabel('Samples');
% visualize a sample input vector
vishdl(5) = figure;
set(gcf, 'color', 'white'); box off; grid off;
start_show_idx = 5; samples_show_num = 10;
window_slide_time = 1; % make sure that is always < IN_SIZE
switch visin.simopts.data.trainvtype
    case 'full'
        plot(visin.netin.raw1, '.b'); hold on;
        plot(visin.netin.raw2, '.g'); box off; grid off;
    case 'interval'
        for idx = 1:samples_show_num
            subplot(1,samples_show_num,idx); set(gcf, 'color', 'white'); box off; grid off;
            plot(visin.netin.trainv1(start_show_idx+idx, :), 'r'); hold on;
            plot(visin.netin.trainv2(start_show_idx+idx, :), 'b');
        end
    case 'sliding'
        for idx = 1:samples_show_num
            subplot(1,samples_show_num,idx); set(gcf, 'color', 'white'); box off; grid off;
            plot(visin.netin.trainv1, 'r'); hold on;
            plot(visin.netin.trainv2, 'b');
        end
end
suptitle('Sample training vectors for input p1 vs p2');

% visualize the adaptive params, net activity and net weights
vishdl(6) = cln_visualize_som(visin, visin.som1); % first som
vishdl(7) = cln_visualize_som(visin, visin.som2); % second som
end