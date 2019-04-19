function [imgs, gt, valid, raw_imgs] = loadTestSample(obj, fidx, varargin)
    opts.read_img = true;
    opts.check_gt_outlier = true;
    opts = vl_argparse(opts, varargin) ;

    imgs = cell(1,4);
    raw_imgs = cell(1,4);
    if opts.read_img
        for i = 1:4
            I = imread(obj.imgs_path{i}{obj.test_idx(fidx)});
            raw_imgs{i} = I;
            if size(I, 3) == 3
                I = rgb2gray(I);
            end
            
            I = single(I);
            if isfield(obj.ocams{i}, 'invisible_mask')
                I(obj.ocams{i}.invisible_mask) = nan;
            end
            I = (I-mean(I(:), 'omitnan')) ./ std(I(:), 'omitnan');
            if isfield(obj.ocams{i}, 'invisible_mask')
                I(obj.ocams{i}.invisible_mask) = 0;
            end
            imgs{i} = gpuArray(I);
        end
    end
    gt = [];
    valid = [];
    if nargout >= 2 && strcmp(obj.dtype, 'gt') && ...
            length(obj.test_depth_path) >= fidx
        gtsize = obj.test_depth_size(fidx,:);
        gt = single(imread((obj.test_depth_path{fidx})));
        gt = obj.sweep.m_d + (gt/100)*(obj.sweep.M_d/655);
        if obj.sweep.imsize(1) < gtsize(1)
           sh = round((gtsize(1)-obj.sweep.imsize(1))/2) +1;
           gt = gt(sh:sh+obj.sweep.imsize(1)-1,:);
        end
        cond = (gt >= obj.sweep.m_d);
        cond2 = imclose(cond, strel('disk', 2));
        if ~opts.check_gt_outlier, cond2 = cond; end
        gt(~cond&~cond2) = obj.sweep.m_d;
        gt = ((gt-obj.sweep.m_d)/obj.sweep.s_d + 1);
        valid = gt >= 1 & gt <= obj.sweep.ndisps;
    end
end