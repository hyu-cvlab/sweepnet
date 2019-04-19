%
% Author: Changhee Won (chwon@hanyang.ac.kr)
%
classdef Database
    %DATABASE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        name
        db_path
        imgs_path
        train_depth_path
        test_depth_path
        train_depth_size % ground-truth depth resolution
        test_depth_size
        train_idx
        test_idx
        train_size % # of train set
        test_size % # of test set
        ocams % cam intrinsics
        poses % 6x4 cam poses
        sweep % spherical sweep parameters struct
        dtype % 'gt' or 'nogt'
        gt_phi % ground-truth phi degree
        opts
    end
    
    methods
        function obj = Database(name, varargin)
            %DATABASE Construct an instance of this class
            %   Detailed explanation goes here
            if isa(name, 'cell')
                tic
                objs = obj.initSequences(name{1}, varargin{:});
                names = [name{1}];
                for i = 2:length(name)
                    tmp = obj.initSequences(name{i}, varargin{:});
                    for j = 1:4
                       objs.imgs_path{j} = [objs.imgs_path{j}; ...
                           tmp.imgs_path{j}];
                    end
                    objs.depth_path = [objs.depth_path; ...
                        tmp.depth_path];
                    ndata = objs.train_size + objs.test_size;
                    objs.train_idx = [objs.train_idx (tmp.train_idx+ndata)];
                    objs.test_idx = [objs.test_idx (tmp.test_idx+ndata)];
                    objs.train_size = objs.train_size + tmp.train_size;
                    objs.test_size = objs.test_size + tmp.test_size;
                    objs.depth_size = [objs.depth_size; tmp.depth_size];
                    objs.dbnames = [objs.dbnames; tmp.dbnames];
                    names = [names '; ' name{i}]; 
                end
                obj = objs;
                fprintf('# Train/Test %d/%d sequences loaded: ', ...
                    obj.train_size, obj.test_size);
                toc
                name = names;
            else
                tic
                obj = obj.initSequences(name, varargin{:});
                fprintf('# Train/Test %d/%d sequences loaded: ', ...
                    obj.train_size, obj.test_size);
                toc
            end
            tic
            obj = obj.initSweep();
            fprintf('# "%s" initilized: ', name);
            toc
            
        end
    end
end

