// array.h
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//         
//

#ifndef ARRAY_H
#define ARRAY_H

#include <mex.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <limits>

#define  PI  3.1415926535897932

#define MAX(a,b) ((a)>(b))? (a) : (b)
#define MIN(a,b) ((a)<(b))? (a) : (b)

template<typename T>
class Array2D {
 public:
  Array2D() {};
  Array2D(int height, int width) {
    Create(height, width);
  }

  Array2D(const mxArray *in) {
    _data = (T*)mxGetData(in);
    const size_t *dims = mxGetDimensions(in);
    _height = dims[0];
    _width = dims[1];
  }

  inline T& operator()(size_t x) { return _data[x]; }
  inline T& operator()(size_t x, size_t y) {
    return _data[x+y*_height]; }
  
  inline T operator()(size_t x) const { return _data[x]; }
  inline T operator()(size_t x, size_t y) const {
    return _data[x+y*_height]; }
  
  inline T* data() const { return _data; }

  inline size_t size() const { return _width*_height; }
  inline size_t width() const { return _width; }
  inline size_t height() const { return _height; }
  inline size_t cols() const { return _width; }
  inline size_t rows() const { return _height; }
  inline void Wrap(mxArray** out) { if (_mxptr) *out = _mxptr; }
  inline void Create(int height, int width) {
    _height = height;
    _width = width;
    int bytes = sizeof(T);
    size_t dims[2] = {(size_t)height, (size_t)width};
    switch (bytes) {
      case 1:
        _mxptr = mxCreateNumericArray(2, dims, mxINT8_CLASS, mxREAL);
        break;
      case 2:
        _mxptr = mxCreateNumericArray(2, dims, mxINT16_CLASS, mxREAL);
        break;
      case 4:
        _mxptr = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
        break;
      case 8:
        _mxptr = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
        break;
      default:;
    };
    _data = (T*)mxGetData(_mxptr);
  }
  inline void Destroy() { if (_mxptr) mxDestroyArray(_mxptr); }
 protected:
  mxArray* _mxptr;
  T* _data;
  int _width, _height;
};

template<typename T>
class Array3D : public Array2D<T> {
 public:
  Array3D() {};
  Array3D(int height, int width, int depth) {
    Create(height, width, depth);
  }

  Array3D(const mxArray *in) {
    this->_data = (T*)mxGetData(in);
    const size_t *dims = mxGetDimensions(in);
    this->_height = dims[0];
    this->_width = dims[1];
    _depth = dims[2];
    _stride2 = this->_width*this->_height;
  }
  inline T& operator()(size_t x) { return this->_data[x]; }
  inline T& operator()(size_t x, size_t y, size_t z) {
    return this->_data[x+y*this->_height+z*_stride2]; }
  
  inline T operator()(size_t x) const { return this->_data[x]; }
  inline T operator()(size_t x, size_t y, size_t z) const {
    return this->_data[x+y*this->_height+z*_stride2]; }

  inline size_t size() const { return this->_width*this->_height*_depth; };
  inline size_t depth() const { return _depth; };
  inline void Create(int height, int width, int depth) {
    this->_width = width;
    this->_height = height;
    _depth = depth;
    _stride2 = width*height;
    int bytes = sizeof(T);
    size_t dims[3] = {(size_t)height, (size_t)width, (size_t)depth};
    
    switch (bytes) {
      case 1:
        this->_mxptr = mxCreateNumericArray(3, dims, mxINT8_CLASS, mxREAL);
        break;
      case 2:
        this->_mxptr = mxCreateNumericArray(3, dims, mxINT16_CLASS, mxREAL);
        break;
      case 4:
        this->_mxptr = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
        break;
      case 8:
        this->_mxptr = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
        break;
      default:;
    };
    this->_data = (T*)mxGetData(this->_mxptr);
  }
 protected:
  int _depth;
  int _stride2;
};

typedef Array2D<float> ArrayFloat;
typedef Array2D<double> ArrayDouble;
typedef Array3D<float> ArrayFloat3D;
typedef Array3D<double> ArrayDouble3D;
#endif