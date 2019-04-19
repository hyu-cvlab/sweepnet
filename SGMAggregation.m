%
% Author: Changhee Won (chwon@hanyang.ac.kr)
%
function [idx, aggr] = SGMAggregation(costs, p1, p2)
%SGMAGGREGATION Summary of this function goes here
%   Detailed explanation goes here
if isa(costs, 'gpuArray')
    costs = single(costs);
    [idx, aggr]= mexGPUSGM(costs, p1, p2);
    idx = gather(idx);
    aggr = gather(aggr);
else
    [aggr] = mexConcurrentSGM(costs, p1, p2);
    [~,idx] = min(aggr, [], 3, 'omitnan');
end
end

