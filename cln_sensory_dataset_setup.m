%% UNSUPERVISED CORRELATION LEARNING NETWORK USING SOM
% Each sensory variable projects onto a SOM network which depending on the
% global network connectivity (1, 2, ... , N vars) connects to other SOM
% associated with other sensory variables.

% For the current implementation we only consider a 2 variable network

% FUNCTION
% Extracts and prepares data for processing from the sensory data file.
%
% ARGS
%   opts - simulation options parametrized for run
%
% RETURN
%   netin - struct with data to be fed in net or for analysis

function netin = cln_sensory_dataset_setup(opts)
        infile     = 'robot_data_jras_paper';        % robot dataset
        data_scaling    = -0.0572957795130823;       % data dependent scaling
        data_freq_samp   = 25;                       % Hz
        trainlen   = opts.data.trainvsize;                           % number of training vectors in the training data set (only for full trainvtype)
switch (opts.data.source)
    case 'sensors'
        sensory_data = load(infile);
        trainlen = length(sensory_data);
        time_units = (1:length(sensory_data))/data_freq_samp;
        % the first sensory variable
        p1 = sensory_data(:, 7)*data_scaling;
    case 'generated'
        % generate some datasets for analysis
        % create first variable between minv and maxv
        num_samples = opts.data.ntrainv;
        time_units = (1:num_samples)/data_freq_samp;
        encoded_val = 4.0;
        p1 = ones(1, num_samples)*encoded_val;
        p2 = p1.*3;
end
% create the second variable depending on the correlation type
switch(opts.data.corrtype)
    case 'algebraic'
        % simple algebraic correlation
        minv = 0; maxv = 0.5;
        p2 = p1.*3;
        for idx=1:num_samples
           additive_noise = minv + (maxv-minv)*rand;
           p1(idx) = p1(idx) + additive_noise;
           additive_noise = minv + (maxv-minv)*rand;
           p2(idx) = p2(idx) + additive_noise;
        end
    case 'temporal'
        % temporal integration
        p2 = zeros(1, length(p1)); p2(1) = p1(1);
        for idx = 2:length(p1)
            p2(idx) = p2(idx-1) + ((1/data_freq_samp)*(p1(idx)));
        end
    case 'nonlinear'
        % nonlinear correlation (trigo func)
        p2 = 4.5*cos(p1./5) + 6.5;
    case 'delay'
        % sine wave and a delayed one
        num_samples = trainlen;
        t = linspace(0,1,num_samples);
        p1 = cos(2*pi*100*t);
        xdft = fft(p1);
        shift = exp(-1j*0.4*pi);
        xdft = xdft.*shift;
        p2 = ifft(xdft,'symmetric');
%          plot(p1,'k'); hold on;
%          plot(p2,'b'); grid on;
%          legend('Original Signal','Delayed Signal');
    case 'amplitude'
        % sine wave and an amplitude modulated one
        num_samples = trainlen;
        t = linspace(0,1,num_samples);
        p1 = cos(2*pi*100*t);
        p2 = 2.5*p1;
%           plot(p1(1:100),'k'); hold on;
%           plot(p2(1:100),'b'); grid on;
%           legend('Original Signal','Modulated Signal');
end
% prepate the training vectors (fixed interval / sliding window /  full dataset)
switch(opts.data.trainvtype) 
    case 'full'
        trainlen = length(p1);
        training_set_size = opts.data.ntrainv;
        training_set_p1 = zeros(training_set_size, trainlen);
        training_set_p2 = zeros(training_set_size, trainlen);
        
        training_set_p1(1, :) = p1;
        training_set_p2(1, :) = p2;
        
        for idx = 2:training_set_size
                training_set_p1(idx, :) = p1 + min(p1./4) + (max(p1./4)-min(p1)./4)*rand;
                training_set_p2(idx, :) = p2 + min(p2./4) + (max(p2./4)-min(p2)./4)*rand;
        end
        
    case 'interval'
        % split the input vectors in training vectors
        training_set_size = round(length(p1)/opts.data.trainvsize-1);
        training_set_p1 = zeros(training_set_size, opts.data.trainvsize);
        training_set_p2 = zeros(training_set_size, opts.data.trainvsize);
        
        training_set_p1(1, :) = p1(1:opts.data.trainvsize);
        training_set_p2(1, :) = p2(1:opts.data.trainvsize);
        
        % fill the training datasets
        for idx = 2:training_set_size
            for jdx = 1:opts.data.trainvsize
                training_set_p1(idx, jdx) = p1(((idx-1)*opts.data.trainvsize + jdx));
                training_set_p2(idx, jdx) = p2(((idx-1)*opts.data.trainvsize + jdx));
            end
        end
    case 'sliding'
        % second we can use a sliding window with a size of trainvsize
        % samples and a time delay of TAU_SLIDE
        
        TAU_SLIDE = 1; % CAREFUL! TAU_SLIDE < opts.data.trainvsize
        
%         training_set_size = round(2*length(p1)/opts.data.trainvsize)*TAU_SLIDE;
%         training_set_p1 = zeros(training_set_size, opts.data.trainvsize);
%         training_set_p2 = zeros(training_set_size, opts.data.trainvsize);
%         
%         training_set_p1(1, :) = p1(1:opts.data.trainvsize);
%         training_set_p2(1, :) = p2(1:opts.data.trainvsize);
%         iidx = 1;
%         % fill the training datasets
%         for idx = 2:(training_set_size)
%             for jdx = 1:opts.data.trainvsize
%                 training_set_p1(idx, jdx) = p1(((iidx)*TAU_SLIDE + jdx));
%                 training_set_p2(idx, jdx) = p2(((iidx)*TAU_SLIDE + jdx));
%             end
%             iidx = iidx + 1;
%             if(((iidx)*TAU_SLIDE + opts.data.trainvsize)>length(p1))
%                 break;
%             end
%         end
        % data to be buffered are p1 and p2
        % trainvsize sample window size with an overlap of TAU_SLIDE samples
        training_set_p1 = buffer(p1, opts.data.trainvsize, TAU_SLIDE)'; 
        training_set_p2 = buffer(p2, opts.data.trainvsize, TAU_SLIDE)'; 
        % remove the first and the last training vectors as they are 0->val
        % and val->0 extensions of the dataset
        training_set_p1(1,:) = training_set_p1(2,:); training_set_p1(end, :) = training_set_p1(2,:);
        training_set_p2(1,:) = training_set_p2(2,:); training_set_p2(end, :) = training_set_p2(2,:);
        [~, training_set_size] = size(training_set_p1');
end
% embed everything in the return struct
netin.raw1 = p1; netin.raw2 = p2;
netin.time = time_units';
netin.trainsetsize = training_set_size;
netin.trainv1 = training_set_p1;
netin.trainv2 = training_set_p2;
fprintf(1, 'cln_sensory_dataset_setup: Created training datasets with %d entries and %d samples.\n', netin.trainsetsize, length(training_set_p1(1,:)));
end