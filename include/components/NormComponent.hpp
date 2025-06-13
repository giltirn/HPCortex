#pragma once
#include <cmath>
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//A component implementing the norm operation on a batch tensor (one for which the last dimension is the batch dimension) along an arbitrary dimension other than the batch dimension
template<typename _FloatType, int TensDim>
class NormComponent{
public:
  typedef _FloatType FloatType;
private:
  int norm_dim;
  FloatType epsilon;

  int in_size[TensDim];
  size_t other_dim_vol;
  size_t stride;
  mutable FLOPScounter value_FLOPS;
  mutable FLOPScounter deriv_FLOPS;
  
  bool setup;

  mutable RingBuffer<Tensor<FloatType,TensDim> > out_buf;
  mutable RingBuffer<Matrix<FloatType> > std_buf;
public:
  
  NormComponent(int norm_dim,FloatType epsilon = 1e-5): norm_dim(norm_dim), epsilon(epsilon), setup(false){
    assert(norm_dim >=0 && norm_dim < TensDim-1);
  }
  NormComponent(const NormComponent &r) = delete;
  NormComponent(NormComponent &&r) = default;
  
  Tensor<FloatType,TensDim> value(const Tensor<FloatType,TensDim> &in);
  
  void deriv(Tensor<FloatType,TensDim> &&dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const;
  
  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return value_or_deriv == 0 ? value_FLOPS.value() : deriv_FLOPS.value(); }
  
  inline void resizeInputBuffer(size_t to){
    out_buf.resize(to);
    std_buf.resize(to);
  }
};

#include "implementation/NormComponent.tcc"
