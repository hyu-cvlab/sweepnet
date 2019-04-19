function [out] = rigidMotion(pts, pose)
%RIGIDMOTION Summary of this function goes here
%   Detailed explanation goes here
if isa(pts, 'gpuArray')
    if length(pose) == 6
        pose = [rodrigues(pose(1:3)) pose(4:6)];
    end
    if ~isa(pose, 'gpuArray')
        pose = gpuArray(single(pose));
    end
    out = mexCUDARigidMotion(pts, pose);
else
    out = rigidmotion(pts, pose);
end
end

