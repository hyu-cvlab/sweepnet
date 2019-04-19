// mexInterp2D.cc
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//         
//
#include <mex.h>
#include "array_util.h"

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  mxClassID type = mxGetClassID(prhs[0]);
  
  if (type == mxDOUBLE_CLASS) {
    ArrayDouble I(prhs[0]);
    ArrayDouble pts(prhs[1]);
    ArrayDouble out(pts.cols(), 1);
    out.Wrap(&plhs[0]);
    Interp2D(I, pts, &out);
  } else if (type == mxSINGLE_CLASS) {
    ArrayFloat I(prhs[0]);
    ArrayFloat pts(prhs[1]);
    ArrayFloat out(pts.cols(), 1);
    out.Wrap(&plhs[0]);
    Interp2D(I, pts, &out);
  }
  
}