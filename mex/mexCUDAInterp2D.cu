// mexCUDAInterp2D.cu
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//         
//

#include <mex.h>
#include <gpu/mxGPUArray.h>
#include "cuda/cudamx.cuh"
#include "cuda/cudamxarray.cuh"
#include "cuda/cudamx_image_util.cuh"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxInitGPU();
  CudaMxArray2D image(prhs[0]), pts(prhs[1]);
  CudaMxArray2D out(pts.width(), 1);
  out.Wrap(&plhs[0]);
  
  Interp2D(image, pts, &out, NAN);

  image.Destroy();
  pts.Destroy();
  out.Destroy();
}