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
%   void

function cln_visualize(visin)
% visualize input data
%----------------------------------------------------------------
figure; 
set(gcf, 'color', 'white');
subplot(2, 1, 1); plot(visin.netin.raw1, 'b'); 
box off; grid off;
ylabel('Input var P1'); xlabel('Samples');
subplot(2, 1, 2);
plot(visin.netin.raw2, 'g'); 
box off; grid off;
ylabel('Input var P2'); xlabel('Samples');
%----------------------------------------------------------------
figure; 
set(gcf, 'color', 'white'); 
box off; grid off;
plot(visin.alphat, '.b'); hold on; 
plot(visin.gammat, '.g'); hold on; 
plot(visin.xit, '.m'); hold on; 
plot(visin.kappat, '.k');
suptitle('Adaptation parameters'); 
xlabel('Epochs');
legend('Learning rate','Cross-modal impact','Inhibitory gain in W update','Hebbian learning rate'); 
%----------------------------------------------------------------
figure; 
set(gcf, 'color', 'white'); 
plot(visin.sigmat, '.r'); 
ylabel('Neighborhood radius');xlabel('Samples');
%----------------------------------------------------------------
% visualize a sample input vector
figure; 
set(gcf, 'color', 'white'); 
box off; grid off;
samples_show_num = 5;
switch visin.simopts.data.trainvtype
    case 'full'
        plot(visin.netin.raw1, 'b'); hold on; 
        plot(visin.netin.raw2, 'g'); 
        box off; grid off;
    case 'interval'
        start_show_idx = 5; 
        for idx = 1:samples_show_num
            subplot(1,samples_show_num,idx); 
            set(gcf, 'color', 'white'); 
            box off; grid off;
            plot(visin.netin.trainv1(start_show_idx + idx, :), 'r'); hold on;
            plot(visin.netin.trainv2(start_show_idx + idx, :), 'b');
        end
    case 'sliding'
        for idx = 1:samples_show_num
            subplot(1,samples_show_num,idx); 
            set(gcf, 'color', 'white'); 
            box off; grid off;
            plot(visin.netin.trainv1(idx, :), 'r'); hold on;
            plot(visin.netin.trainv2(idx, :), 'b');
        end
end
suptitle('Sample training vectors for input p1 vs p2');
%----------------------------------------------------------------
% visualize the adaptive params, net activity and net weights
cln_visualize_som(visin, visin.som1); % first som
cln_visualize_som(visin, visin.som2); % second som
end