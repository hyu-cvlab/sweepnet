%
% Author: Changhee Won (chwon@hanyang.ac.kr)
%
function initMatlabPath(dir_matconvnet)

if nargin < 1
    dir_matconvnet = '../matconvnet-1.0-beta25';
end
addpath(genpath('.'));
cur = pwd;
cd(dir_matconvnet);
addpath(genpath(fullfile(dir_matconvnet, 'contrib')));
run matlab/vl_setupnn;
cd(cur);

end
