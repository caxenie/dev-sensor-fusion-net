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
if (strcmp(visin.simopts.data.trainvtype,'hunt')==1)
    figure;
    set(gcf, 'color', 'white');
    subplot(2, 1, 1); plot(visin.netin.raw1, 'b');
    box off; grid off;
    ylabel('Input dataset var P1'); xlabel('Samples');
    subplot(2, 1, 2);
    plot(visin.netin.raw2, 'r');
    box off; grid off;
    ylabel('Input dataset var P2'); xlabel('Samples');
    
    figure;
    set(gcf, 'color', 'white');
    subplot(2, 1, 1); plot(visin.netin.trainv1(:,1), 'b'); hold on;
    plot(visin.netin.trainv1(:,2), 'b');
    box off; grid off;
    ylabel('Input dataset var P1 trainv'); xlabel('Samples');
    subplot(2, 1, 2);
    plot(visin.netin.trainv2(:,1), 'r');  hold on;
    plot(visin.netin.trainv2(:,2), 'r');
    box off; grid off;
    ylabel('Input dataset var P2 trainv'); xlabel('Samples');
else
    %----------------------------------------------------------------
    figure;
    set(gcf, 'color', 'white');
    subplot(2, 1, 1); plot(visin.netin.raw1, 'b');
    box off; grid off;
    ylabel('Input dataset var P1'); xlabel('Samples');
    subplot(2, 1, 2);
    plot(visin.netin.raw2, 'r');
    box off; grid off;
    ylabel('Input dataset var P2'); xlabel('Samples');
end
%----------------------------------------------------------------
figure;
set(gcf, 'color', 'white');
box off; grid off;
plot(visin.alphat, '.b'); hold on;
if(strcmp(visin.simopts.net.xmodlearn, 'none')~=1)
    plot(visin.gammat, '.g'); hold on;
    plot(visin.xit, '.m'); hold on;
    plot(visin.kappat, '.k'); box off;
    suptitle('Adaptation parameters');
    xlabel('Epochs');
    legend('Learning rate','Cross-modal impact','Inhibitory gain in W update','Hebbian learning rate');
else
    xlabel('Epochs'); box off;
    legend('Learning rate');
end
%----------------------------------------------------------------
figure;
set(gcf, 'color', 'white');
plot(visin.sigmat, '.k'); box off;
ylabel('Neighborhood radius');xlabel('Samples');
%----------------------------------------------------------------
% visualize a sample input vector only if not correlation hunting mode is
% enable in the main simulation
if(strcmp(visin.simopts.data.trainvtype, 'hunt')==0)
    figure;
    set(gcf, 'color', 'white');
    box off; grid off;
    samples_show_num = 5;
    switch visin.simopts.data.trainvtype
        case 'full'
            plot(visin.netin.raw1, 'b'); hold on;
            plot(visin.netin.raw2, 'r'); grid off;
        case 'interval'
            start_show_idx = 5;
            for idx = 1:samples_show_num
                subplot(1,samples_show_num,idx);
                set(gcf, 'color', 'white'); grid off;
                plot(visin.netin.trainv1(start_show_idx + idx, :), 'r'); hold on;
                plot(visin.netin.trainv2(start_show_idx + idx, :), 'b'); box off;
            end
        case 'sliding'
            for idx = 1:samples_show_num
                subplot(1,samples_show_num,idx);
                set(gcf, 'color', 'white'); grid off;
                plot(visin.netin.trainv1(idx, :), 'r'); hold on;
                plot(visin.netin.trainv2(idx, :), 'b'); box off;
            end
    end
    suptitle('Sample training vectors for input p1 vs p2');
end
%----------------------------------------------------------------
% visualize the adaptive params, net activity and net weights
cln_visualize_som(visin, visin.som1); % first som
cln_visualize_som(visin, visin.som2); % second som
end