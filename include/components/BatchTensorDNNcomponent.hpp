#pragma once
#include <type_traits>
#include <sstream>
#include <ModelConfig.hpp>
#include <Tensors.hpp>
#include <Buffers.hpp>
#include <Linalg.hpp>

//A component implementing    W_{ij} X_{..., j, ...,  b} + B_j    where W is a weight matrix and X is a tensor of at least dimension 2. The last dimension is always assumed to be the batch dimension
template<typename Config, int TensDim, typename ActivationFunc>
class BatchTensorDNNcomponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  Matrix<FloatType> weights;
  Vector<FloatType> bias;
  
  int batch_size;
  int contract_dim;
  bool use_bias;
 
  int in_dims[TensDim];
  int out_dims[TensDim];
  size_t other_size; //volume of dimensions other than batch_dim and contract_dim
  size_t stride;
  mutable FLOPScounter value_FLOPS;
  mutable FLOPScounter deriv_FLOPS;
  bool setup;

  ActivationFunc activation_func;
  
  //Storage from last call to "value"
  //Buffer size > 1 depending on rank if doing pipelining
  mutable BufferType<Tensor<FloatType,TensDim> > in_buf;
  mutable BufferType<Tensor<FloatType,TensDim> > activation_deriv_buf;
public:

  
  BatchTensorDNNcomponent(const Matrix<FloatType> &weights, const Vector<FloatType> &bias, int contract_dim, const ActivationFunc &activation):
    weights(weights), bias(bias), use_bias(true),setup(false), activation_func(activation), contract_dim(contract_dim)
  {
    assert(bias.size(0) == weights.size(0));
  }
  BatchTensorDNNcomponent(const Matrix<FloatType> &_weights, int contract_dim, const ActivationFunc &activation):
    weights(_weights), bias(_weights.size(0),0.), use_bias(false), setup(false), activation_func(activation), contract_dim(contract_dim)
  { }
  
  BatchTensorDNNcomponent(const BatchTensorDNNcomponent &r) = delete;
  BatchTensorDNNcomponent(BatchTensorDNNcomponent &&r) = default;
  
  //Forward pass
  Tensor<FloatType,TensDim> value(const Tensor<FloatType,TensDim> &x, EnableDeriv enable_deriv = DerivNo);

  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&dCost_by_dOut, Tensor<FloatType,TensDim> &dCost_by_dIn) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  size_t FLOPS(int value_or_deriv) const{ return value_or_deriv == 0 ? value_FLOPS.value() : deriv_FLOPS.value(); }
  
  //accumulated #params for layers here and below
  inline int nparams() const{ return weights.size(0)*weights.size(1) + (use_bias ? bias.size(0) : 0);  }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off) const;

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    in_buf.resize(to);
    activation_deriv_buf.resize(to);
  }
};

#include "implementation/BatchTensorDNNcomponent.tcc"
