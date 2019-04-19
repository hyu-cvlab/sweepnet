// array_util.h
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//         
//

#ifndef ARRAY_UTIL_H
#define ARRAY_UTIL_H

#include "array.h"

template <typename T> inline
T Interp2D(const Array2D<T>& I, int x, int y) {
  T xp = x - 1; // C indexing
  T yp = y - 1;
  int w = I.width();
  int h = I.height();
  if (xp >= w || xp < 0 || yp >= h || yp < 0) {
    return mxGetNaN();
  }

  const int x0 = int(xp);
  const int y0 = int(yp);
  const int x1 = x0 < (w-1) ? x0 + 1 : x0;
  const int y1 = y0 < (h-1) ? y0 + 1 : y0;
  const T rx = xp - x0;
  const T ry = yp - y0;

  T interp = (I(y0,x0)*(1-ry) + I(y1,x0)*ry) * (1-rx) +
                 (I(y0,x1)*(1-ry) + I(y1,x1)*ry) * rx;
  return interp;
}

// 2 x N,
template <typename T> inline
void Interp2D(const Array2D<T>& I, const Array2D<T>& pts, 
              Array2D<T> *out) {
                
  for (int n = 0; n < pts.cols(); n++) {
    (*out)(n) = Interp2D(I, pts(0,n), pts(1,n));
  }
}

#endif