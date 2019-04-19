// mexConcurrentSGM.cc
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//         
//

#include "array.h"
#include "thread.h"

using namespace std;

const int rowdiffs[8] = { 0, -1,  1, -1,  0,  1,  1, -1};
const int coldiffs[8] = {-1,  0, -1, -1,  1,  0,  1,  1};

template<typename T>
class AggregateThread : public cvlib::Thread {
 public:
  const Array3D<T>& cost_volume;
  size_t h, w, ndisps, path;
  T p1, p2;
  Array2D<T> min_costs;
  Array2D<int> min_ds;

// out
  Array3D<T>* aggrs;

  AggregateThread (const Array3D<T>& cost_volume, T p1, T p2, int path,
                  Array3D<T>* aggrs)
    : cost_volume(cost_volume), p1(p1), p2(p2), path(path), aggrs(aggrs) {
    h = cost_volume.height();
    w = cost_volume.width();
    ndisps = cost_volume.depth();
    min_costs.Create(h,w);
    min_ds.Create(h,w);
  }

  ~AggregateThread() {
    min_costs.Destroy();
    min_ds.Destroy();
  }

  void Aggregate(int i, int j, int qi, int qj) {
    bool is_border = (qi < 0 || qj < 0 || qi >= h || qj >= w);
    T min_prev_cost;
    int min_prev_d;
    if (!is_border) {
      min_prev_cost = min_costs(qi, qj);
      min_prev_d = min_ds(qi, qj);
    }
    T min_cost = 1e9;
    int min_d;
    for (int d = 0; d < ndisps; d++) {
      T cost = cost_volume(i,j,d);
      if (mxIsNaN(cost)) {
        (*aggrs)(i,j,d) = mxGetNaN();
        continue;
      }
      if (!is_border) {
        T prev_cost = (*aggrs)(qi, qj, d);
        T prev_plus = 1e9, prev_minus = 1e9;
        if (d < ndisps - 1) {
          prev_plus = (*aggrs)(qi, qj, d+1);
        } 
        if (d > 0) {
          prev_minus = (*aggrs)(qi, qj, d-1);
        }

        if (abs(d-min_prev_d) <= 1) {
          cost += min(prev_cost, min(
            prev_plus, prev_minus)+p1) - min_prev_cost;
        } else {
          cost += min(prev_cost, min(min_prev_cost+p2, min(
            prev_plus, prev_minus)+p1)) - min_prev_cost;
        }
      }
      (*aggrs)(i,j,d) = cost;
      if (cost < min_cost) {
        min_cost = cost;
        min_d = d;
      }
    }
    min_costs(i,j) = min_cost;
    min_ds(i,j) = min_d;
  }


  void LeftUpperAggregation() {
    for (int j = 0; j < w; j++) {
      int qj = j + coldiffs[path];
      for (int i = 0; i < h; i++) {
        int qi = i + rowdiffs[path];
        Aggregate(i, j, qi, qj);
      }
    }
  }

  void RightLowerAggregation() {
    for (int j = w - 1; j >= 0; j--) {
      int qj = j + coldiffs[path];
      for (int i = h - 1; i >= 0; i--) {
        int qi = i + rowdiffs[path];
        Aggregate(i, j, qi, qj);
      }
    }
  }

private:
  void ThreadWork() {
    if (path < 4) {
      LeftUpperAggregation();      
    } else {
      RightLowerAggregation();
    }
  };
};


template<typename T>
void SGMAggregate(const Array3D<T>& cost_volume, double p1, double p2,
                  mxArray** out_ptr) {
  int h = cost_volume.height();
  int w = cost_volume.width();
  int d = cost_volume.depth();
  Array3D<T> aggr_volume(h,w,d);
  aggr_volume.Wrap(out_ptr);

  Array3D<T> aggrs[8];
  AggregateThread<T> *aggr_threads[8];

  for (int i = 0; i< 8; i++) {
    aggrs[i].Create(h, w, d);
    aggr_threads[i] = new AggregateThread<T>(
      cost_volume, p1, p2, i, &aggrs[i]);
    aggr_threads[i]->Create();
  }

  for (int i = 0; i < 8; i++) {
    aggr_threads[i]->Join();
    for (int n = 0; n < aggr_volume.size(); n++) {
      aggr_volume(n) += aggrs[i](n);
    }
    aggrs[i].Destroy();
    delete aggr_threads[i];
  }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // input
  mxClassID type = mxGetClassID(prhs[0]);
  double p1 = mxGetScalar(prhs[1]);
  double p2 = mxGetScalar(prhs[2]);
  if (type == mxDOUBLE_CLASS) {
    ArrayDouble3D cost_volume(prhs[0]);
    SGMAggregate<double>(cost_volume, p1, p2, &plhs[0]);
  } else {
    ArrayFloat3D cost_volume(prhs[0]);
    SGMAggregate<float>(cost_volume, p1, p2, &plhs[0]);
  }

}