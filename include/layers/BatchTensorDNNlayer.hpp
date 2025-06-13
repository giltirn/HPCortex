#pragma once
#include "LayerCommon.hpp"
#include <components/BatchTensorDNNcomponent.hpp>

//A layer implementing    W_{ij} X_{..., j, ...,  b} + B_j    where W is a weight matrix and X is a tensor of at least dimension 2. The last dimension is always assumed to be the batch dimension
template<typename _FloatType, int TensDim, typename _InputType, typename Store, typename ActivationFunc>
class BatchTensorDNNlayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  Store leaf;
  BatchTensorDNNcomponent<FloatType,TensDim,ActivationFunc> cpt;
public:
  typedef LeafTag tag;

  //With bias
  BatchTensorDNNlayer(Store &&leaf, const Matrix<FloatType> &weights, const Vector<FloatType> &bias, int contract_dim, const ActivationFunc &activation): cpt(weights,bias,contract_dim,activation), leaf(std::move(leaf))
  {  }
  //Without bias (i.e. linear layer)
  BatchTensorDNNlayer(Store &&leaf, const Matrix<FloatType> &weights, int contract_dim, const ActivationFunc &activation): cpt(weights,contract_dim,activation), leaf(std::move(leaf))
  {  }

  BatchTensorDNNlayer(const BatchTensorDNNlayer &r) = delete;
  BatchTensorDNNlayer(BatchTensorDNNlayer &&r) = default;
  
  //Forward pass
  Tensor<FloatType,TensDim> value(const InputType &x);

  int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  int update(int off, const Vector<FloatType> &new_params);
  
  int step(int off, const Vector<FloatType> &derivs, FloatType eps);

  size_t FLOPS(int value_or_deriv) const{ return cpt.FLOPS(value_or_deriv) + leaf.v.FLOPS(value_or_deriv); }
  
  //accumulated #params for layers here and below
  inline int nparams() const{ return cpt.nparams() + leaf.v.nparams(); }

  //off measured from *end*, return new off
  int getParams(Vector<FloatType> &into, int off);

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    cpt.resizeInputBuffer(to);
    leaf.v.resizeInputBuffer(to);
  }

};

#define LAYER_TYPE BatchTensorDNNlayer<FLOATTYPE(U),TensDim,INPUTTYPE(U),DDST(u),ActivationFunc>
template<int TensDim, typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto batch_tensor_dnn_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights, const Vector<FLOATTYPE(U)> &bias, int contract_dim, const ActivationFunc &activation)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), weights, bias, contract_dim, activation);
}
template<int TensDim, typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto batch_tensor_dnn_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights, int contract_dim, const ActivationFunc &activation)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), weights, contract_dim, activation);
}

//default initialization of weights of size fan_out x fan_in using glorotUniformRandom   and bias of size fan_out to zeros.
template<int TensDim, typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto batch_tensor_dnn_layer(U &&u, int contract_dim, int fan_out, int fan_in, const ActivationFunc &activation)-> LAYER_TYPE{
  Matrix<FLOATTYPE(U)> weights(fan_out, fan_in);
  glorotUniformRandom(weights);
  Vector<FLOATTYPE(U)> bias(fan_out, 0.);  
  return LAYER_TYPE(std::forward<U>(u), weights, bias, contract_dim, activation);
}

//default initialization of weights of size fan_out x fan_in using glorotUniformRandom   and *no bias*
template<int TensDim, typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto batch_tensor_unbiased_dnn_layer(U &&u, int contract_dim, int fan_out, int fan_in, const ActivationFunc &activation)-> LAYER_TYPE{
  Matrix<FLOATTYPE(U)> weights(fan_out, fan_in);
  glorotUniformRandom(weights);
  return LAYER_TYPE(std::forward<U>(u), weights, contract_dim, activation);
}

#undef LAYER_TYPE


#define LAYER_TYPE BatchTensorDNNlayer<FLOATTYPE(U),2,INPUTTYPE(U),DDST(u),ActivationFunc>
template<typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights,const Vector<FLOATTYPE(U)> &bias, const ActivationFunc &activation){
  return LAYER_TYPE(std::forward<U>(u), weights, bias, 0, activation);
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights,const Vector<FLOATTYPE(U)> &bias){
  return dnn_layer(std::forward<U>(u),weights,bias,noActivation<FLOATTYPE(U)>());
}

//default initialization of weights of size fan_out x fan_in using glorotUniformRandom   and bias of size fan_out to zeros.
template<typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, int fan_out, int fan_in, const ActivationFunc &activation){
  Matrix<FLOATTYPE(U)> weights(fan_out, fan_in);
  glorotUniformRandom(weights);
  Vector<FLOATTYPE(U)> bias(fan_out, 0.);   
  return LAYER_TYPE(std::forward<U>(u), weights, bias, 0, activation);
}
#undef LAYER_TYPE

#include "implementation/BatchTensorDNNlayer.tcc"
