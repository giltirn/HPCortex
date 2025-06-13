#pragma once
#include <cmath>
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <Linalg.hpp>

//A component implementing single element slice for batch tensors along a specific dimension, reducing the tensor dimension by 1
//eg  B_{ijk} = C_{ijlk}
template<typename _FloatType, int TensDim>
class BatchTensorDimensionSliceComponent{
public:
  typedef _FloatType FloatType;
private:
  int slice_dim;
  int slice_idx;
  
  int in_size[TensDim];
  int out_size[TensDim-1];
  
  size_t other_dim_vol;
  size_t offset_in;
  bool setup;
public:

  BatchTensorDimensionSliceComponent(int slice_dim, int slice_idx): slice_dim(slice_dim), slice_idx(slice_idx), setup(false){
    assert(slice_dim >= 0 && slice_dim < TensDim-1);
  }
  BatchTensorDimensionSliceComponent(const BatchTensorDimensionSliceComponent &r) = delete;
  BatchTensorDimensionSliceComponent(BatchTensorDimensionSliceComponent &&r) = default;
  
  Tensor<FloatType,TensDim-1> value(const Tensor<FloatType,TensDim> &in);
  
  void deriv(Tensor<FloatType,TensDim-1> &&dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const;

  size_t FLOPS(int value_or_deriv) const{ return 0; }
  
  inline int nparams() const{ return 0; }
};

#include "implementation/BatchTensorDimensionSliceComponent.tcc"
