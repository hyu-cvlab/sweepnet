%
% Author: Changhee Won (chwon@hanyang.ac.kr)
%
function obj = initSequences(obj, name, varargin)
%INITSEQUENCES Summary of this function goes here
%   Detailed explanation goes here
obj.name = name;
obj.imgs_path = cell(1,4);
% set default arguments
opts.imsize = [160 640];
opts.ndisps = 192;
opts.phi = 45;
opts.max_fov = 0.4; % Fisheye fov threshold; % about 220~226 degree
opts.root_path = './';
opts.sweep = true;
switch lower(name)
case 'sunny'
    opts.db_path = fullfile(opts.root_path, 'sunny');
    opts.min_depth = 55;
    opts.max_depth = 1/eps;
    opts.img_fmt = '%04d.png';

    obj.train_idx = 1:700;
    obj.test_idx = 701:1000;
    obj.dtype = 'gt';
    obj.gt_phi = 45;
case 'cloudy'
    opts.db_path = fullfile(opts.root_path, 'cloudy');
    opts.min_depth = 55;
    opts.max_depth = 1/eps;
    opts.img_fmt = '%04d.png';

    obj.train_idx = 1:700;
    obj.test_idx = 701:1000;
    obj.dtype = 'gt';
    obj.gt_phi = 45;
case 'sunset'
    opts.db_path = fullfile(opts.root_path, 'sunset');
    opts.min_depth = 55;
    opts.max_depth = 1/eps;
    opts.img_fmt = '%04d.png';

    obj.train_idx = 1:700;
    obj.test_idx = 701:1000;
    obj.dtype = 'gt';
    obj.gt_phi = 45;
case 'realindoorsample'
    opts.db_path = fullfile(opts.root_path, 'real_indoor_sample');
    opts.min_depth = 550;
    opts.max_depth = 1/eps;
    opts.img_fmt = '%04d.png';

    obj.train_idx = [];
    obj.test_idx = 1:3;
    obj.dtype = 'nogt';
    obj.gt_phi = 45;

otherwise
    error('failed to get DB: "%s"\n', name);
end
opts = vl_argparse(opts, varargin);
obj.opts = opts;
obj.db_path = opts.db_path;
obj.sweep.imsize = opts.imsize;
obj.sweep.min_depth = opts.min_depth;
obj.sweep.max_depth = opts.max_depth;
obj.sweep.ndisps = opts.ndisps;
obj.sweep.phi = opts.phi;
obj.sweep.ndisps = opts.ndisps;
obj.sweep.load_grid = opts.sweep;
temp = load(fullfile(obj.db_path, 'intrinsic_extrinsic.mat'));
obj.ocams = temp.ocams;
obj.poses = temp.poses;
obj.sweep.center = single(mean(temp.poses(4:6,:),2));
obj.train_size = size(obj.train_idx, 2);
obj.test_size = size(obj.test_idx, 2);

% Load images path
for i = 1:4
    imgs = dir(fullfile(obj.db_path, num2str(i, 'cam%d/*.png')));
    obj.imgs_path{i} = cellfun(@fullfile, {imgs(:).folder}', {imgs(:).name}', ...
        'UniformOutput', false);
end
% Load depths path
if strcmp(obj.dtype, 'gt')
    train_depth_path = fullfile(obj.db_path, ...
        num2str(opts.imsize(2), 'depth_train_%d'));
    train_depths = dir(fullfile(train_depth_path, '*.png'));
    obj.train_depth_path = cellfun(@fullfile, {train_depths(:).folder}', ...
        {train_depths(:).name}', 'UniformOutput', false);
    
    test_depth_path = fullfile(obj.db_path, ...
        num2str(opts.imsize(2), 'depth_test_%d'));
    test_depths = dir(fullfile(test_depth_path, '*.png'));
    obj.test_depth_path = cellfun(@fullfile, {test_depths(:).folder}', ...
        {test_depths(:).name}', 'UniformOutput', false);
    gw = obj.sweep.imsize(2);
    gh = round(gw / (180 / obj.gt_phi));
    obj.train_depth_size = repmat([gh gw], length(obj.train_depth_path), 1);
    obj.test_depth_size = repmat([gh gw], length(obj.test_depth_path), 1);
%     obj.dbnames = repmat({obj.name}, length(obj.depth_path), 1);
end
    
end

