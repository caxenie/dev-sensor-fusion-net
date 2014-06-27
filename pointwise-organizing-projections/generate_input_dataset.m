% function to generate the input dataset
% currently a mixture of k Gaussians is used
function out = generate_input_dataset(k, sigma0, n, type)
switch type
    case 'gauss-mixture'
        % generates a mixture of K Gaussians with standard deviation sigma
        % and centered on dk
        out.K = k;
        out.sigmat = sigma0;
        out.d = zeros(1, out.K);
        for kdx = 1:out.K
            out.d(kdx) = round(1 + (n - 1)*rand);
        end
        out.d = sort(out.d);
        for idx =1:n
            sum_gauss = 0.0;
            for kdx = 1:out.K
                sum_gauss = sum_gauss + exp(-(idx-out.d(kdx))^2/(out.sigmat)^2);
            end
            out.data(idx) =  sum_gauss;
        end
end
end
