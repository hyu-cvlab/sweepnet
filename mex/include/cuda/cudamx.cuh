// cudamx.cuh
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//

#ifndef _CUDA_MX_CUH
#define _CUDA_MX_CUH

#define BLOCK2D 16
#define BLOCK1D 1024
#define BLOCK3D 8
#define WARP_SIZE 32

inline int NumGrid(int n, int block_size) {
  int v = n / block_size;
  return (n % block_size == 0) ? v : v+1;
}

// colum major 2d array
template <typename T> __global__ 
void _Assign2D(T* des, T* src, const int w, const int h) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < w && j < h) {
    int idx = j*w + i;
    des[idx] = src[idx];
  }
}

template <typename T> __global__ 
void _Mult2D(T* arr, const T scalar, const int w, const int h) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < w && j < h) {
    int idx = j*w + i;
    arr[idx] = arr[idx] * scalar;
  }
}

template <typename T> __global__ 
void _Assign2D(T* arr, const T scalar, const int w, const int h) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < w && j < h) {
    int idx = j*w + i;
    arr[idx] = scalar;
  }
}

template <typename T> __global__
void _Add2D(T* arr, const T scalar, const int w, const int h) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < w && j < h) {
    int idx = j*w + i;
    arr[idx] = arr[idx] + scalar;
  }
}

template <typename T> __global__
void _Add2D(T* lhs, T* rhs, const int w, const int h) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < w && j < h) {
    int idx = j*w + i;
    lhs[idx] = lhs[idx] + rhs[idx];
  }
}

// column, row, depth major 3d array

template <typename T> __global__ 
void _Assign3D(T* des, T* src, const int w, const int h, const int d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i < w && j < h && k < d) {
    int idx = k*w*h + j*w + i;
    des[idx] = src[idx];
  }
}

template <typename T> __global__ 
void _Mult3D(T* arr, const T scalar, const int w, const int h, const int d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i < w && j < h && k < d) {
    int idx = k*w*h + j*w + i;
    arr[idx] = arr[idx] * scalar;
  }
}

template <typename T> __global__ 
void _Assign3D(T* arr, const T scalar, const int w, const int h, const int d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i < w && j < h && k < d) {
    int idx = k*w*h + j*w + i;
    arr[idx] = scalar;
  }
}

template <typename T> __global__
void _Add3D(T* arr, const T scalar, const int w, const int h, const int d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i < w && j < h && k < d) {
    int idx = k*w*h + j*w + i;
    arr[idx] = arr[idx] + scalar;
  }
}

template <typename T> __global__
void _Add3D(T* lhs, T* rhs, const int w, const int h, const int d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i < w && j < h && k < d) {
    int idx = k*w*h + j*w + i;
    lhs[idx] = lhs[idx] + rhs[idx];
  }
}

template <typename T> __global__
void _Add3D(T* a, T* b, T* out, const int w, const int h, const int d) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i < w && j < h && k < d) {
    int idx = k*w*h + j*w + i;
    out[idx] = a[idx] + b[idx];
  }
}

#endif

