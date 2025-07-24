#pragma once
#include <cmath>
#include <type_traits>
#include <sstream>
#include <ModelConfig.hpp>
#include <Tensors.hpp>
#include <Buffers.hpp>
#include <Linalg.hpp>

//A component implementing the softmax operation over the rows of a batched matrix ( Tensor<FloatType,3> with the last dimension the batch dimension ) with optional masking
//softmax(In + M)  where M is zero (no masking) or is *strictly* upper triangular (diagonal elements also zero) with nonzero elements =-inf
//If masking, the input must be a square
template<typename Config>
class BatchedMatrixRowSoftMaxComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  FloatType beta;
  mutable BufferType<Tensor<FloatType,3> > out_buf;
  mutable FLOPScounter value_FLOPS;
  mutable FLOPScounter deriv_FLOPS;
  bool use_mask;
public:
  
  BatchedMatrixRowSoftMaxComponent(bool use_mask = false, FloatType beta = 1.0): use_mask(use_mask), beta(beta){ }
  BatchedMatrixRowSoftMaxComponent(const BatchedMatrixRowSoftMaxComponent &r) = delete;
  BatchedMatrixRowSoftMaxComponent(BatchedMatrixRowSoftMaxComponent &&r) = default;
  
  Tensor<FloatType,3> value(const Tensor<FloatType,3> &in) const;
  void deriv(Tensor<FloatType,3> &&dcost_by_dOut, Tensor<FloatType,3> &dcost_by_dIn) const;

  size_t FLOPS(int value_or_deriv) const{ return value_or_deriv == 0 ? value_FLOPS.value() : deriv_FLOPS.value(); }
  
  inline int nparams() const{ return 0; }

  inline void resizeInputBuffer(size_t to){
    out_buf.resize(to);
  }
};

#include "implementation/BatchedMatrixRowSoftMaxComponent.tcc"
