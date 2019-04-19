// cudamx_cudnn.cuh
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//

#ifndef _CUDA_MX_CUDNN_CUH
#define _CUDA_MX_CUDNN_CUH

#include <cudnn.h>
#include "cudamx.cuh"
#include "cudamxarray.cuh"


#define CHECKCUDNN(expression)                               \
{                                                          \
  cudnnStatus_t status = (expression);                     \
  if (status != CUDNN_STATUS_SUCCESS) {                    \
    mexErrMsgIdAndTxt("CUDNN:ERROR",                                 \
      "Error on line %d: %s", __LINE__, cudnnGetErrorString(status));    \
  }                                                        \
}

#endif