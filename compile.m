%
% Author: Changhee Won (chwon@hanyang.ac.kr)
%
function compile()
outdir = 'build';
include = '-I./include/';
if ~exist('./mex/build', 'file')
   mkdir('./mex/build'); 
end
compileCPU(outdir, include);
compileCuda(outdir, include);
end

function compileCPU(outdir, include)


cd './mex'
try
    mex(include, 'mexInterp2D.cc', '-outdir', outdir);
    mex(include, 'mexConcurrentSGM.cc', '-outdir', outdir);
catch ME
    cd '../'
    error(ME.message)
end
cd '../'
end

function compileCuda(outdir, include)
% cudnn_dir = '-L/usr/local/cuda/lib64';
cd './mex'
try
    mexcuda('mexCUDAInterp2D.cu', include, '-outdir', outdir, ...
        '-lstdc++', '-lc');
    mexcuda('mexGPUSGM.cu', include, '-outdir', outdir, ...
        '-lstdc++', '-lc');
catch ME
    cd '../'
    error(ME.message)
end
cd '../'

end