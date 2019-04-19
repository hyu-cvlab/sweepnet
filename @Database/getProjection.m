%
% Author: Changhee Won (chwon@hanyang.ac.kr)
%
function [projection, pc] = getProjection(obj, imgs, res)
    %GETPROJECTION Summary of this function goes here
    %   Detailed explanation goes here
    switch class(obj.sweep.rays)
        case 'single', cast = @(z) single(gather(z)) ;
        case 'double', cast = @(z) double(gather(z)) ;
    end

    depth = 1./(obj.sweep.m_d + (res(:)-1)*obj.sweep.s_d);
    P = repmat(cast(depth(:)'),3,1).*obj.sweep.rays + ...
        repmat(obj.sweep.center, 1, length(depth));

    projections = cell(1,4);
    counts = zeros(size(res));
    I_sum = zeros(size(res));
    for i = 1:4
        P2 = rigidMotion(P, invext(obj.poses(:,i)));
        p = world2cam_fast(P2, obj.ocams{i}, obj.opts.max_fov);
        p(isnan(p)) = -2;
        projections{i} = interp2D(cast(imgs{i}), p);
        valid = ~isnan(projections{i});
        I_sum(valid) = I_sum(valid) + projections{i}(valid);
        counts(valid) = counts(valid) + 1;
    end
    proj = (I_sum./counts);
    projection = makeGrayMap(proj, min(proj(:)), max(proj(:)));
    pc = pointCloud(P', 'Color', repmat(projection(:), 1, 3));
end

