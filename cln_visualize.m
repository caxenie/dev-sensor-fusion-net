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
title('Input var P1');
subplot(2, 1, 2);plot(visin.netin.raw2, '.g'); box off; grid off;
title('Input var P2'); xlabel('Samples');
% visualize cross-correlation
vishdl(2) = figure;
set(gcf, 'color', 'white');
plot(xcorr(visin.netin.raw1, visin.netin.raw2), '.r'); box off; grid off;
title('Xcorr analysis'); xlabel('Samples');
% visualize a sample input vector
vishdl(3) = figure;
switch visin.simopts.data.trainvtype
    case 'interval'
        start_show_idx = 10; samples_show_num = 10;
        for idx = 1:samples_show_num
            subplot(1,samples_show_num,idx); set(gcf, 'color', 'white'); box off; grid off;
            plot(visin.netin.trainv1(start_show_idx+idx, :), 'r'); hold on;
            plot(visin.netin.trainv2(start_show_idx+idx, :), 'b');
        end
    case 'sliding'
        start_show_idx = 10; samples_show_num = 10; window_slide_time = 10; % CAREFUL ! window_slide_time < IN_SIZE
        for idx = 1:samples_show_num
            subplot(1,samples_show_num,idx); set(gcf, 'color', 'white'); box off; grid off;
            plot(visin.netin.trainv1(start_show_idx+idx*window_slide_time, :), 'r'); hold on;
            plot(visin.netin.trainv2(start_show_idx+idx*window_slide_time, :), 'b');
        end
end
title('Sample training vectors for input p1 vs p2');

% visualize the adaptive params, net activity and net weights
vishdl(4) = cln_visualize_som(visin, visin.som1); % first som
vishdl(5) = cln_visualize_som(visin, visin.som2); % second som
end