# SweepNet
This repository contains Matlab codes for paper, "SweepNet:Wide-baseline Omnidirectional Depth Estimation" (ICRA 2019). [[arxiv](https://arxiv.org/abs/1902.10904)]

**Contact**: Changhee Won (changhee.1.won@gmail.com)


## Prerequisites
### List of code/library dependencies
- MATLAB R2017b, 2018a, 2018b, 2019a
- CUDA 9.0, 10.0
- [MatConvNet](http://www.vlfeat.org/matconvnet/) 1.0-beta25


## How to run
### Compile mex files (c++, cuda)
```
>> compile
(output:
  ./mex/build/mexConcurrentSGM.mex*
  ./mex/build/mexGPUSGM.mex*
  ./mex/build/mexInterp2D.mex*
  ./mex/build/mexCUDAInterp2D.mex*)
```

### Test (run_test_sweepnet.m)
- Pretrained weights: [[sweepnet_sunny_14.mat](http://bit.ly/2UuaEiE)]
- Set arguments in the script
 - "matconvnet_path": path to matconvnet
 - "pretrained_model": path to pretrained_weights
 - "data_opts.db_path": path to dataset

- Run

```
>> run_test_sweepnet
```

### Example result
<img src="https://user-images.githubusercontent.com/7540390/56401334-3d51db00-6293-11e9-94ae-c9679d1773b5.png" width=50%>

## Dataset
You can download the synthetic datasets in the [project page](http://cvlab.hanyang.ac.kr/project/omnistereo).

The directory structure should be like this:
```
sunny/
     /cam1/
     /cam2/
     /cam3/
     /cam4/
     /depth_train_640/
     /intrinsic_extrinsic.mat
```

## Citation
Will be updated soon
```
```