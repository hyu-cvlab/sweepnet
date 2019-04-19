// cudamx_image_util.cuh
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//

#ifndef _CUDA_MX_IMAGE_UTIL_CUH
#define _CUDA_MX_IMAGE_UTIL_CUH

#include "cuda/cudamx.cuh"
#include "cuda/cudamxarray.cuh"

__global__
void _Interp2D(const CudaMxArray2D image, const CudaMxArray2D pts2d,
                CudaMxArray2D out, const float blank=NAN) {

  int w = image.width();
  int h = image.height();
  int w_target = out.width();
  int h_target = out.height();

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (j < w_target && i < h_target) {
    int n = j*h_target + i;
    float x = pts2d(0,n) - 1;
    float y = pts2d(1,n) - 1;
    if (x < 0 || x > w-1 || y < 0 || y > h-1 || isnan(x) || isnan(y)) {
      out(i, j) = blank;
    } else {
      const int x0 = static_cast<int>(x);
      const int y0 = static_cast<int>(y);
      int x1 = x0 < w ? x0 + 1 : x0;
      int y1 = y0 < h ? y0 + 1 : y0;
      
      const float rx = x - x0, ry = y - y0;
      out(i, j) = static_cast<float>(
          (image(y0, x0) * (1 - rx) + image(y0, x1) * rx) * (1 - ry) +
          (image(y1, x0) * (1 - rx) + image(y1, x1) * rx) * ry);
    }
  }
}


inline __host__
void Interp2D(const CudaMxArray2D& image, CudaMxArray2D& pts2d,
                CudaMxArray2D* out, const float blank=0,
                const int block_size=BLOCK2D) {
  _Interp2D<<<NumGrid(out->height(), block_size), block_size>>>(
                                                  image, pts2d, *out, blank);
}

template <typename T> inline __device__
void RigidMotion(const T* pts3d, const CudaMxArray2D& pose,
                const int i, T* out) {
  double x = pts3d[3*i];
  double y = pts3d[3*i+1];
  double z = pts3d[3*i+2];
  out[3*i] = pose(0,0)*x+pose(0,1)*y+pose(0,2)*z + pose(0,3);
  out[3*i+1] = pose(1,0)*x+pose(1,1)*y+pose(1,2)*z + pose(1,3);
  out[3*i+2] = pose(2,0)*x+pose(2,1)*y+pose(2,2)*z + pose(2,3);
}

inline __device__
void RigidMotion(const CudaMxArray2D& pts3d, const CudaMxArray2D& pose,
                const int i, CudaMxArray2D* out) {
  double x = pts3d(0,i);
  double y = pts3d(1,i);
  double z = pts3d(2,i);
  (*out)(0,i) = pose(0,0)*x+pose(0,1)*y+pose(0,2)*z + pose(0,3);
  (*out)(1,i) = pose(1,0)*x+pose(1,1)*y+pose(1,2)*z + pose(1,3);
  (*out)(2,i) = pose(2,0)*x+pose(2,1)*y+pose(2,2)*z + pose(2,3);
}

__global__ // call from host
void _RigidMotion(const CudaMxArray2D pts3d, const CudaMxArray2D pose,
                CudaMxArray2D out) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < pts3d.width()) RigidMotion(pts3d, pose, i, &out);
}

template <typename T>  __global__ // call from device
void _RigidMotion(const T* pts3d, const CudaMxArray2D& pose,
                const int cols, T* out) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < cols) RigidMotion(pts3d, pose, i, out);
}

__host__
void RigidMotion(const CudaMxArray2D& pts3d, const CudaMxArray2D& pose,
                CudaMxArray2D* out, int block_size=BLOCK1D) {
  int n = pts3d.width();
  _RigidMotion<<<NumGrid(n, block_size), block_size>>>(pts3d, pose, *out);
}

#endif