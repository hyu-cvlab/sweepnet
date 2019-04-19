// cudamx_array.cuh
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//

#ifndef _CUDA_MX_ARRAY_CUH
#define _CUDA_MX_ARRAY_CUH

#include <mex.h>
#include <gpu/mxGPUArray.h>

// !!!!!!ONLY FLOAT ARRAYS!!!!!!!!!!!!!!

class CudaMxArray2D {
 public:
  __host__ __device__
  CudaMxArray2D(): _height(0), _width(0) {};

  __host__
  CudaMxArray2D(const mxArray* in) {
    _gpu = const_cast<mxGPUArray*>(mxGPUCreateFromMxArray(in));
    const size_t *dims = mxGPUGetDimensions(_gpu);
    _data = (float *)mxGPUGetData(_gpu);
    _height = dims[0];
    
    if (mxGPUGetNumberOfDimensions(_gpu) >= 2) 
      _width = dims[1];
    else
      _width = 1;
  }

  __host__
  CudaMxArray2D(const size_t height, const size_t width,
              mxGPUInitialize init=MX_GPU_DO_NOT_INITIALIZE) {
    Create(height, width, init);
  }

  __host__ __device__
  CudaMxArray2D(const CudaMxArray2D& in)
  : _width(in.width()), _height(in.height()), _data(in.data()), _gpu(in.gpu()) { }

  inline __device__ float& operator()(size_t x) { return _data[x]; }
  inline __device__ float& operator()(size_t x, size_t y) {
    return _data[x+y*_height]; }
  
  inline __device__ float operator()(size_t x) const { return _data[x]; }
  inline __device__ float operator()(size_t x, size_t y) const {
    return _data[x+y*_height]; }
  
  inline __host__ __device__ float* data() const { return _data; }
  inline __host__ __device__ mxGPUArray* gpu() const { return _gpu; }
  inline __host__ __device__ size_t size() const { return _width*_height; };
  inline __host__ __device__ size_t width() const { return _width; }
  inline __host__ __device__ size_t height() const { return _height; }
  inline __host__ __device__ size_t cols() const { return _width; }
  inline __host__ __device__ size_t rows() const { return _height; }


  inline __host__ void Wrap(mxArray **out) {
    *out = mxGPUCreateMxArrayOnGPU(_gpu);
  }

  inline __host__ void Create(const size_t height, const size_t width,
                          mxGPUInitialize init=MX_GPU_DO_NOT_INITIALIZE) {
    const size_t dims[] = {height, width};
    _gpu = mxGPUCreateGPUArray(2, dims, mxSINGLE_CLASS, mxREAL, init);
    _data = (float *)mxGPUGetData(_gpu);
    _height = height;
    _width = width;
  }

  inline __host__ void Destroy() { mxGPUDestroyGPUArray(_gpu); }
  inline __host__
  CudaMxArray2D& operator<<(const float scalar) {
    int block_size = BLOCK2D;
    dim3 grid(NumGrid(_height, block_size), NumGrid(_width, block_size));
    dim3 block(block_size, block_size);
    _Assign2D<float><<<grid, block>>>(_data, scalar, _height, _width);
    return *this;
  }
  inline __host__
  CudaMxArray2D& operator+=(const CudaMxArray2D& rhs) {
    int block_size = BLOCK2D;
    dim3 grid(NumGrid(_height, block_size), NumGrid(_width, block_size));
    dim3 block(block_size, block_size);
    _Add2D<float><<<grid, block>>>(_data, rhs.data(), _height, _width);
    return *this;
  }

  
 protected:
  size_t _width, _height;
  mxGPUArray *_gpu;
  float *_data;
};

class CudaMxArray3D : public CudaMxArray2D {
 public:
  __host__ __device__
  CudaMxArray3D(): CudaMxArray2D(), _depth(0) {};

  __host__
  CudaMxArray3D(const mxArray* in) :CudaMxArray2D(in) {
    if (mxGPUGetNumberOfDimensions(_gpu) >= 3) {
      const size_t *dims = mxGPUGetDimensions(_gpu);
      _depth = dims[2];
    } else {
      _depth = 1;
    }
  }

  __host__
  CudaMxArray3D(const size_t height, const size_t width,
                const size_t depth,
                mxGPUInitialize init=MX_GPU_DO_NOT_INITIALIZE) {
    Create(height, width, depth, init);
  }

  inline __device__ float& operator()(size_t x, size_t y, size_t z) {
    return _data[x+y*_height+z*_height*_width]; }

  inline __device__ float operator()(size_t x, size_t y, size_t z) const {
    return _data[x+y*_height+z*_height*_width]; }

  inline __host__ __device__ size_t size() const {
    return _width*_height*_depth; }

  inline __host__ __device__ size_t depth() const { return _depth; }

  inline __host__ void Create(const size_t height, const size_t width,
                              const size_t depth, 
                              mxGPUInitialize init=MX_GPU_DO_NOT_INITIALIZE) {
    const size_t dims[] = {height, width, depth};
    _gpu = mxGPUCreateGPUArray(3, dims, mxSINGLE_CLASS, mxREAL, init);
    _data = (float *)mxGPUGetData(_gpu);
    _height = height;
    _width = width;
    _depth = depth;
  }

  inline __host__
  CudaMxArray3D& operator+=(const CudaMxArray3D& rhs) {
    int block_size = BLOCK3D;
    dim3 grid(NumGrid(this->_height, block_size),
      NumGrid(this->_width, block_size), NumGrid(_depth, block_size));
    dim3 block(block_size, block_size, block_size);
    _Add3D<float><<<grid, block>>>(this->_data, rhs.data(),
      this->_height, this->_width, this->_depth);
    return *this;
  }

  inline __host__
  CudaMxArray3D& operator<<(const float scalar) {
    int block_size = BLOCK3D;
    dim3 grid(NumGrid(this->_height, block_size),
      NumGrid(this->_width, block_size), NumGrid(_depth, block_size));
    dim3 block(block_size, block_size, block_size);
    _Assign3D<float><<<grid, block>>>(this->_data,
      scalar, this->_height, this->_width, _depth);
    return *this;
  }

 protected:
  size_t _depth;
};

class CudaMxArray4D : public CudaMxArray3D {
 public:
  __host__ __device__
  CudaMxArray4D(): CudaMxArray3D(), _channel(0) {};

  __host__
  CudaMxArray4D(const mxArray* in): CudaMxArray3D(in) {
    if (mxGPUGetNumberOfDimensions(_gpu) >= 4) {
      const size_t *dims = mxGPUGetDimensions(_gpu);
      _channel = dims[3];
    } else {
      _channel = 1;
    }
  }

  __host__
  CudaMxArray4D(const size_t height, const size_t width,
                const size_t depth, const size_t channel,
                mxGPUInitialize init=MX_GPU_DO_NOT_INITIALIZE) {
    Create(height, width, depth, channel, init);
  }

  inline __device__ float& operator()(size_t x, size_t y, size_t z, size_t c) {
    return _data[x+y*_height+z*_height*_width+c*_height*_width*_depth]; }

  inline __device__ float operator()(size_t x, size_t y, size_t z, size_t c)
    const { 
      return _data[x+y*_height+z*_height*_width+c*_height*_width*_depth]; 
    }

  inline __host__ __device__ size_t size() const {
    return _width*_height*_depth*_channel; }

  inline __host__ __device__ size_t channel() const { return _channel; }

  inline __host__ void Create(const size_t height, const size_t width,
                              const size_t depth, const size_t channel,
                              mxGPUInitialize init=MX_GPU_DO_NOT_INITIALIZE) {
    const size_t dims[] = {height, width, depth, channel};
    _gpu = mxGPUCreateGPUArray(4, dims, mxSINGLE_CLASS, mxREAL, init);
    _data = (float *)mxGPUGetData(_gpu);
    _height = height;
    _width = width;
    _depth = depth;
    _channel = channel;
  }

 protected:
  size_t _channel;
};

class CudaMxArray5D : public CudaMxArray4D {
 public:
  __host__ __device__
  CudaMxArray5D(): CudaMxArray4D() {};

  __host__
  CudaMxArray5D(const mxArray* in) :CudaMxArray4D(in) {
    if (mxGPUGetNumberOfDimensions(_gpu) >= 5) {
      const size_t *dims = mxGPUGetDimensions(_gpu);
      _channel2 = dims[4];
    } else {
      _channel2 = 1;
    }
  }

  __host__
  CudaMxArray5D(const size_t height, const size_t width,
                const size_t depth, const size_t channel,
                const size_t channel2,
                mxGPUInitialize init=MX_GPU_DO_NOT_INITIALIZE) {
    Create(height, width, depth, channel, channel2, init);
  }

  inline __device__ float& operator()(size_t x, size_t y, size_t z, 
                                      size_t c, size_t c2) {
    return _data[x+y*_height+z*_height*_width+c*_height*_width*_depth+
      c2*_height*_width*_depth*_channel]; }

  inline __device__ float operator()(size_t x, size_t y, size_t z,
                                    size_t c, size_t c2) const { 
      return _data[x+y*_height+z*_height*_width+c*_height*_width*_depth+
        c2*_height*_width*_depth*_channel]; 
    }

  inline __host__ __device__ size_t size() const {
    return _width*_height*_depth*_channel*_channel2; }

  inline __host__ __device__ size_t channel2() const { return _channel2; }

  inline __host__ void Create(const size_t height, const size_t width,
                              const size_t depth, const size_t channel,
                              const size_t channel2,
                              mxGPUInitialize init=MX_GPU_DO_NOT_INITIALIZE) {
    const size_t dims[] = {height, width, depth, channel, channel2};
    _gpu = mxGPUCreateGPUArray(5, dims, mxSINGLE_CLASS, mxREAL, init);
    _data = (float *)mxGPUGetData(_gpu);
    _height = height;
    _width = width;
    _depth = depth;
    _channel = channel;
    _channel2 = channel2;
  }

 protected:
  size_t _channel2;

};

#endif