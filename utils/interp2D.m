function [out] = interp2D(I, pts, shape)
%INTERP2D Summary of this function goes here
%   Detailed explanation goes here
if isa(I, 'gpuArray')
    out = mexCUDAInterp2D(I, pts-1);
else
    out = mexInterp2D(I, pts);
end

if nargin > 2
    out = reshape(out, shape);
end
end

