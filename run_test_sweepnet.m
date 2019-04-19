%
% Author: Changhee Won (chwon@hanyang.ac.kr)
%
%% Initialize
clear;
matconvnet_path = '../matconvnet-1.0-beta25';
initMatlabPath(matconvnet_path);

% Load pretrained_weights;
pretrained_model = 'sweepnet_sunny_14.mat';
[feature_net, matching_net] = loadSweepNet(pretrained_model);

% Load database
data_opts.phi = 45;
data_opts.imsize = [300 1200];
data_opts.db_path = './real_indoor_sample';
data = Database('RealIndoorSample', data_opts);
% Example for Sunny DB
% data_opts.db_path = 'path_to_sunny/sunny';
% data = Database('Sunny', data_opts);

% SGM params
p1 = 0.1;
p2 = 12.0;

%% SweepNet

idxs1 = [1 2 3 4];
idxs2 = [2 3 4 1];

costs = gpuArray(zeros([data.sweep.imsize data.sweep.ndisps], 'single'));
cost = gpuArray(zeros([data.sweep.imsize 4], 'single'));
figure;
for fidx = 1:data.test_size
    [imgs, gt, valid] = data.loadTestSample(fidx);
    fprintf('# Proccessing %04d: ', fidx);
    tic
    for d = 1:data.sweep.ndisps
        sweeps = cell(1,4);
        invalids = cell(1,4);
        for i = 1:4
            sweeps{i} = vl_nnbilinearsampler(imgs{i}, ...
                gpuArray(data.sweep.grids{i}(:,:,:,d)));
            % add circular pad
            sweeps{i} = [sweeps{i}(:,end-1:end) sweeps{i} sweeps{i}(:,1:2)];
            invalids{i} = squeeze(data.sweep.grids{i}(1,:,:,d) < -1);
        end
        sweeps = {'input_1', sweeps{1}, 'input_2', sweeps{2}, ...
            'input_3', sweeps{3}, 'input_4', sweeps{4}};
        
        % Extract unary features
        feature_net.eval(sweeps);
        clear sweeps;
        
        % Compute matching costs
        for i = 1:4
           prefix = 'conv18_relu_';
           inputs = {'feat_1', reshape(cat(2, ...
               feature_net.getVar([prefix num2str(idxs1(i))]).value, ...
               feature_net.getVar([prefix num2str(idxs2(i))]).value),...
               [data.sweep.imsize/2 64])};
           
           matching_net.eval(inputs);

           invalid = invalids{idxs1(i)} | invalids{idxs2(i)};
           matching_net.vars(end).value(invalid) = nan;
           cost(:,:,i) = matching_net.vars(end).value;
        end
        costs(:,:,d) = mean(cost, 3, 'omitnan');
    end
    % Cost aggregation by SGM
    [inv_idx, ~] = SGMAggregation(costs, p1, p2);
    toc
    % Visualize
    proj = data.getProjection(imgs, inv_idx);
    vis = [repmat(proj, [1 1 3]); disp_to_color(inv_idx)];
    title_str = sprintf('DB: %s #%04d', data.name, fidx);
    if ~isempty(gt)
        err = abs(inv_idx-gt)/data.sweep.ndisps*100;
        mae = mean(err(valid));
        vis = [vis; makeColorMap(err,0,10)]; 
        title_str = [title_str num2str(mae, ', MAE: %.2f')];
    end
    clf; imshow(vis);
    title(title_str);
    pause
end


