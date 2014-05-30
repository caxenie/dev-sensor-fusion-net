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
switch (opts.data.source)
    case 'sensors'
        % noise params (standard deviation)
        minv = 0; maxv = 2;
        % specific params for sensor data
        infile     = 'robot_data_jras_paper';                   % robot dataset
        data_scaling    = -0.0572957795130823;                  % data dependent scaling
        data_freq_samp   = 25;                                  % Hz
        sensory_data = load(infile);                            % get the data
        num_samples = length(sensory_data);                        % size of the input vector
        time_units = (1:num_samples)/data_freq_samp;               % time units for display
        p1 = sensory_data(:, 7)*data_scaling;                   % generate the first variable (sample rate of change data from gyro)
    case 'generated'
        % noise params (standard deviation)
        minv = 1; maxv = 0.01;
        % bounds for the input values in p1 and p2
        minp = 0; maxp = 15;
        % specific params for artificial data
        num_samples = opts.data.numsamples;                     % number of samples in the full dataset (similar to sensors)
        data_freq_samp = 25;                                    % artificial data sampling freq
        time_units = (1:num_samples)/data_freq_samp;            % time units for display
        p1 = zeros(1, num_samples);
        for idx = 1:num_samples
            p1(idx) = minp + (maxp - minp)*rand;                % generate the first variable
        end
        switch(opts.data.corrtype)
            case 'delay'
                % sine wave and a delayed one + noise
                t = linspace(0,1,num_samples);
                p1 = cos(2*pi*100*t);
                xdft = fft(p1);
                shift = exp(-1j*0.4*pi);
                xdft = xdft.*shift;
                p2 = ifft(xdft,'symmetric');
                % add some noise over the signals
                for idx=1:num_samples
                    additive_noise = minv + (maxv-minv)*rand;
                    p1(idx) = p1(idx) + additive_noise;
                    additive_noise = minv + (maxv-minv)*rand;
                    p2(idx) = p2(idx) + additive_noise;
                end
            case 'amplitude'
                % sine wave and an amplitude modulated one + noise
                t = linspace(0,1,num_samples);
                p1 = cos(2*pi*100*t);
                p2 = 3*p1;
                % add some noise over the signals
                for idx=1:num_samples
                    additive_noise = minv + (maxv-minv)*rand;
                    p1(idx) = p1(idx) + additive_noise;
                    additive_noise = minv + (maxv-minv)*rand;
                    p2(idx) = p2(idx) + additive_noise;
                end
        end
end
% create the second variable depending on the correlation type
switch(opts.data.corrtype)
    case 'algebraic'
        % simple algebraic correlation
        p2 = p1.*3;
        % add some noise over the signals
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
        % add some noise over the signals
        for idx=1:num_samples
            additive_noise = minv + (maxv-minv)*rand;
            p1(idx) = p1(idx) + additive_noise;
            additive_noise = minv + (maxv-minv)*rand;
            p2(idx) = p2(idx) + additive_noise;
        end
    case 'nonlinear'
        % nonlinear correlation (trigo func)
        p2 = 3.0*cos(p1./3) + 3.0;
        % add some noise over the signals
        for idx=1:num_samples
            additive_noise = minv + (maxv-minv)*rand;
            p1(idx) = p1(idx) + additive_noise;
            additive_noise = minv + (maxv-minv)*rand;
            p2(idx) = p2(idx) + additive_noise;
        end
end
% prepate the training vectors (fixed interval / sliding window /  full dataset)
switch(opts.data.trainvtype)
    case 'full'
        % init
        trainlen = length(p1);
        training_set_size = opts.data.ntrainv;
        training_set_p1 = zeros(training_set_size, trainlen);
        training_set_p2 = zeros(training_set_size, trainlen);
        training_set_p1(1, :) = p1;
        training_set_p2(1, :) = p2;
        % create signals and add some noise
        for idx = 2:training_set_size
            training_set_p1(idx, :) = p1 + min(p1./3) + (max(p1./3)-min(p1)./3)*rand;
            training_set_p2(idx, :) = p2 + min(p2./3) + (max(p2./3)-min(p2)./3)*rand;
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
        % samples and a time delay of tau_slide
        tau_slide = opts.data.slidesize; % CAREFUL! tau_slide < opts.data.trainvsize
        % data to be buffered are p1 and p2
        % trainvsize sample window size with an overlap of tau_slide samples
        training_set_p1 = buffer(p1, opts.data.trainvsize, tau_slide)';
        training_set_p2 = buffer(p2, opts.data.trainvsize, tau_slide)';
        % remove the first and the last training vectors as they are 0->val
        % and val->0 extensions of the dataset as buffer() adds itseld
        % padding to the given data
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