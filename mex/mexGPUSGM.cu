// mexGPUSGM.cu
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//         
//
#include <mex.h>
#include <gpu/mxGPUArray.h>
#include "cuda/cudamx.cuh"
#include "cuda/cudamxarray.cuh"
#include "cuda/cudamx_image_util.cuh"

const int rowdiffs[8] = { 0,  1, -1, -1,  0,  1, -1,  1};
const int coldiffs[8] = {-1, -1, -1,  0,  1,  1,  1,  0};
// left, left, left, up, right, right, right, down

__global__
void _Permute(const CudaMxArray3D src, CudaMxArray3D des, int s) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  int w = src.width();
  int h = src.height();
  int d = src.depth();
  if (i < h && j < w && k < d) {
    if (s == 0)
      des(k,i,j) = src(i,j,k);
    else
      des(j,k,i) = src(i,j,k);
  }
}

void Permute(const CudaMxArray3D& src, CudaMxArray3D* des, int s) {
  int block_size = BLOCK3D;
  dim3 grid(NumGrid(src.height(), block_size),
    NumGrid(src.width(), block_size), NumGrid(src.depth(), block_size));
  dim3 block(block_size, block_size, block_size);
  _Permute<<<grid, block>>>(src, *des, s);
}

__device__ inline
void Aggregate(const CudaMxArray3D& cost_volume,
              int i, int j, int d, const int qi, const int qj,
              const float p1, const float p2,
              CudaMxArray2D *min_costs, CudaMxArray2D* min_ds,
              CudaMxArray2D *cost_cache, CudaMxArray2D *idx_cache,
              CudaMxArray3D* aggregated_volume) {
  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int w = cost_volume.depth();
  int h = cost_volume.width();
  int ndisps = cost_volume.height();
  bool is_border = (qi < 0 || qj < 0 || qj >= h || qi >= w);
  __shared__ float cdata[BLOCK2D][BLOCK2D];
  __shared__ float idata[BLOCK2D][BLOCK2D];
  float cost = cost_volume(d,j,i);
  if (isnan(cost)) {
    (*aggregated_volume)(d,j,i) = NAN;
    cdata[tid_x][tid_y] = 1e9;
    idata[tid_x][tid_y] = d;
  } else {
    if (!is_border) {
      float min_prev_cost = (*min_costs)(qj, qi);
      float min_prev_d = (*min_ds)(qj, qi);
      float prev_cost = (*aggregated_volume)(d,qj,qi);
      float prev_plus = 1e9, prev_minus = 1e9;
      if (d < ndisps - 1) prev_plus = (*aggregated_volume)(d+1,qj,qi);
      if (d > 0) prev_minus = (*aggregated_volume)(d-1,qj,qi);

      if (fabs(d-min_prev_d) <= 1) {
        cost += min(prev_cost, min(
          prev_plus, prev_minus)+p1) - min_prev_cost;
      } else {
        cost += min(prev_cost, min(min_prev_cost+p2, min(
          prev_plus, prev_minus)+p1)) - min_prev_cost;
      }
    }
    (*aggregated_volume)(d,j,i) = cost;
    cdata[tid_x][tid_y] = cost;
    idata[tid_x][tid_y] = d;
  }
  __syncthreads();

  for (int k = blockDim.y/2; k>=1; k/=2) {
    if (tid_y < k && d+k < ndisps) {
      if (cdata[tid_x][tid_y] > cdata[tid_x][tid_y+k]) {
        cdata[tid_x][tid_y] = cdata[tid_x][tid_y+k];
        idata[tid_x][tid_y] = idata[tid_x][tid_y+k];
      }
    }
    __syncthreads();
  }
  if (tid_y == 0) {
    if (w == cost_cache->height()) {
      (*cost_cache)(i, blockIdx.y) = cdata[tid_x][0];
      (*idx_cache)(i, blockIdx.y) = idata[tid_x][0];
    } else {
      (*cost_cache)(j, blockIdx.y) = cdata[tid_x][0];
      (*idx_cache)(j, blockIdx.y) = idata[tid_x][0];
    }
  }
  __syncthreads();
}


__global__
void GetMinCost(CudaMxArray2D cost_cache,
                CudaMxArray2D idx_cache,
                int idx,
                CudaMxArray2D min_costs,
                CudaMxArray2D min_ds) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int tid = threadIdx.y; 

  if (i < cost_cache.height()) {
    for (int k = blockDim.y/2; k>=1; k/=2) {
      if (tid < k && tid+k < cost_cache.width()) {
        if (cost_cache(i, tid) > cost_cache(i, tid+k)) {
          cost_cache(i, tid) = cost_cache(i, tid+k);
          idx_cache(i, tid) = idx_cache(i, tid+k);
        }
      }
      __syncthreads();
    }
    if (tid == 0) {
      if (cost_cache.height() == min_costs.height()) {
        min_costs(i,idx) = cost_cache(i,0);
        min_ds(i,idx) = idx_cache(i,0);
      } else {
        min_costs(idx,i) = cost_cache(i,0);
        min_ds(idx,i) = idx_cache(i,0);
      }
    }
  }
}

__global__
void ColAggr(const CudaMxArray3D cost_volume,
              const int rowdiff, const int coldiff,
              const float p1, const float p2,
              int i,
              CudaMxArray2D min_costs,
              CudaMxArray2D min_ds,
              CudaMxArray2D cost_cache,
              CudaMxArray2D idx_cache,
              CudaMxArray3D aggregated_volume) {
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  int k = blockIdx.y*blockDim.y + threadIdx.y;
  int h = cost_volume.width();
  int w = cost_volume.depth();
  int d = cost_volume.height();
  if (j < h && k < d) {
    int qi = i + coldiff;
    int qj = j + rowdiff;
    Aggregate(cost_volume, i, j, k, qi, qj, p1, p2, &min_costs, &min_ds,
      &cost_cache, &idx_cache, &aggregated_volume);
  }
}

__global__
void RowAggr(const CudaMxArray3D cost_volume,
              const int rowdiff, const int coldiff,
              const float p1, const float p2,
              int j,
              CudaMxArray2D min_costs,
              CudaMxArray2D min_ds,
              CudaMxArray2D cost_cache,
              CudaMxArray2D idx_cache,
              CudaMxArray3D aggregated_volume) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int k = blockIdx.y*blockDim.y + threadIdx.y;
  int h = cost_volume.width();
  int w = cost_volume.depth();
  int d = cost_volume.height();
  if (i < w && k < d) {
    int qi = i + coldiff;
    int qj = j + rowdiff;
    Aggregate(cost_volume, i, j, k, qi, qj, p1, p2, &min_costs, &min_ds,
      &cost_cache, &idx_cache, &aggregated_volume);
  }
}

__global__
void TEST(CudaMxArray3D final_volume, CudaMxArray2D cost_cache) {
  for (int n = 0; n < 12; n++) {
    final_volume(n,63,0) = cost_cache(63,n);
  }
}


void SGMAggregate(const CudaMxArray3D& cost_volume, double p1, double p2,
                  CudaMxArray3D* final_volume) {
  int h = cost_volume.width();
  int w = cost_volume.depth();
  int ndisps = cost_volume.height();
  
  CudaMxArray3D aggregated_volume(ndisps, h, w);
  CudaMxArray2D min_costs(h, w);
  CudaMxArray2D min_ds(h, w);
  
  int block_size = BLOCK2D;
  int nblock_disp = NumGrid(ndisps, block_size);
  dim3 grid_w(NumGrid(w, block_size), nblock_disp);
  dim3 grid_h(NumGrid(h, block_size), nblock_disp);
  dim3 block(block_size, block_size);

  CudaMxArray2D cost_cache;
  CudaMxArray2D idx_cache;
  
  for (int path_idx = 0; path_idx < 8; path_idx++) {
    min_costs << 1e9;
    min_ds << 0;
    if (path_idx > 6) {
      cost_cache.Create(w, nblock_disp);
      idx_cache.Create(w, nblock_disp);
      for (int j = h-1; j >= 0; j--) { // down
        RowAggr<<<grid_w, block>>>(
          cost_volume, rowdiffs[path_idx],
          coldiffs[path_idx], p1, p2, j, min_costs, min_ds,
          cost_cache, idx_cache, aggregated_volume);
        dim3 grid(NumGrid(w, block_size), 1);
        dim3 block_cache(block_size, nblock_disp);
        GetMinCost<<<grid, block_cache>>>(cost_cache, idx_cache,
          j, min_costs, min_ds);
      }
    } else if (path_idx > 3) {
      cost_cache.Create(h, nblock_disp);
      idx_cache.Create(h, nblock_disp);
      for (int i = w-1; i >= 0; i--) { // right
        ColAggr<<<grid_h, block>>>(
          cost_volume, rowdiffs[path_idx],
          coldiffs[path_idx], p1, p2, i, min_costs, min_ds,
          cost_cache, idx_cache, aggregated_volume);
        dim3 grid(NumGrid(h, block_size), 1);
        dim3 block_cache(block_size, nblock_disp);
        GetMinCost<<<grid, block_cache>>>(cost_cache, idx_cache,
          i, min_costs, min_ds);
      }
    } else if (path_idx > 2) { //up
      cost_cache.Create(w, nblock_disp);
      idx_cache.Create(w, nblock_disp);
      for (int j = 0; j < h; j++) {
        RowAggr<<<grid_w, block>>>(
          cost_volume, rowdiffs[path_idx],
          coldiffs[path_idx], p1, p2, j, min_costs, min_ds,
          cost_cache, idx_cache, aggregated_volume);
        dim3 grid(NumGrid(w, block_size), 1);
        dim3 block_cache(block_size, nblock_disp);
        GetMinCost<<<grid, block_cache>>>(cost_cache, idx_cache,
          j, min_costs, min_ds);
      }
    } else { // left
      cost_cache.Create(h, nblock_disp);
      idx_cache.Create(h, nblock_disp);
      for (int i = 0; i < w; i++) {
        ColAggr<<<grid_h, block>>>(
          cost_volume, rowdiffs[path_idx],
          coldiffs[path_idx], p1, p2, i, min_costs, min_ds,
          cost_cache, idx_cache, aggregated_volume);
        dim3 grid(NumGrid(h, block_size), 1);
        dim3 block_cache(block_size, block_size);
        GetMinCost<<<grid, block_cache>>>(cost_cache, idx_cache,
          i, min_costs, min_ds);
      } 
    }
    (*final_volume) += aggregated_volume;
    cost_cache.Destroy();
    idx_cache.Destroy();
  }
  aggregated_volume.Destroy();
  min_costs.Destroy();
  min_ds.Destroy();
}

__global__
void FindDispIdx(CudaMxArray3D cost_volume, CudaMxArray2D idx) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int ndisps = cost_volume.height();
  int h = cost_volume.width();
  int w = cost_volume.depth();

  if (i < h && j < w) {
    float min_cost = 1e9;
    int min_d = 0;

    for (int d = 0; d < ndisps; d++) {
      float cost = cost_volume(d, i, j);
      if (!isnan(cost) && cost < min_cost) {
        min_cost = cost;
        min_d = d;
      }
    }
    idx(i,j) = min_d + 1; // matlab indexing
  }
} 


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  mxInitGPU();
  CudaMxArray3D cost_volume(prhs[0]);
  double p1 = mxGetScalar(prhs[1]);
  double p2 = mxGetScalar(prhs[2]);

  int h = cost_volume.height();
  int w = cost_volume.width();
  int ndisps = cost_volume.depth();

  CudaMxArray3D aggr_volume(h,w,ndisps);
  CudaMxArray2D final_idx(h,w);
  final_idx.Wrap(&plhs[0]);
  aggr_volume.Wrap(&plhs[1]);
  
  CudaMxArray3D cost_volume_perm(ndisps, h, w);
  CudaMxArray3D aggr_volume_perm(ndisps, h, w);
  Permute(cost_volume, &cost_volume_perm, 0);

  aggr_volume_perm << 0;
  SGMAggregate(cost_volume_perm, p1, p2, &aggr_volume_perm);

  dim3 grid_idx(NumGrid(h, BLOCK2D), NumGrid(w, BLOCK2D));
  dim3 block_idx(BLOCK2D, BLOCK2D);
  FindDispIdx<<<grid_idx, block_idx>>>(aggr_volume_perm, final_idx);
  Permute(aggr_volume_perm, &aggr_volume, 1);

  cost_volume.Destroy();
  cost_volume_perm.Destroy();
  aggr_volume.Destroy();
  aggr_volume_perm.Destroy();
  final_idx.Destroy();
}