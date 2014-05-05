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
        START_IDX = 13; SAMPLE_DATA_CHUNKS = 10;
        for idx = 1:SAMPLE_DATA_CHUNKS
            subplot(1,SAMPLE_DATA_CHUNKS,idx); set(gcf, 'color', 'white'); box off; grid off;
            plot(visin.netin.trainv1(START_IDX+idx, :), 'r'); hold on;
            plot(visin.netin.trainv2(START_IDX+idx, :), 'b');
        end
    case 'sliding'
        START_IDX = 300; SAMPLE_DATA_CHUNKS = 10; TAU_SLIDE = 10; % CAREFUL ! TAU_SLIDE < IN_SIZE
        for idx = 1:SAMPLE_DATA_CHUNKS
            subplot(1,SAMPLE_DATA_CHUNKS,idx); set(gcf, 'color', 'white'); box off; grid off;
            plot(visin.netin.trainv1(START_IDX+idx*TAU_SLIDE, :), 'r'); hold on;
            plot(visin.netin.trainv2(START_IDX+idx*TAU_SLIDE, :), 'b');
        end
end
title('Sample training vectors for input p1 vs p2');

% visualize the adaptive params, net activity and net weights
vishdl(4) = cln_visualize_som(visin, visin.som1); % first som
vishdl(5) = cln_visualize_som(visin, visin.som2); % second som
end