#pragma once
#include "LayerCommon.hpp"
#include <components/MatrixTensorContractComponent.hpp>

//A layer implementing    W_{ij} X_{..., j, b}   where W is a weight matrix and X is a tensor of at least dimension 2. The last dimension is always assumed to be the batch dimension
template<typename _FloatType, int TensDim, typename _InputType, typename Store>
class MatrixTensorContractLayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  Store leaf;
  MatrixTensorContractComponent<FloatType,TensDim> cpt;
public:
  typedef LeafTag tag;
  
  MatrixTensorContractLayer(Store &&leaf, const Matrix<FloatType> &weights): cpt(weights), leaf(std::move(leaf))
  {  }
  MatrixTensorContractLayer(const MatrixTensorContractLayer &r) = delete;
  MatrixTensorContractLayer(MatrixTensorContractLayer &&r) = default;
  
  //Forward pass
  Tensor<FloatType,TensDim> value(const InputType &x);

  int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  int update(int off, const Vector<FloatType> &new_params);
  
  int step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return cpt.nparams() + leaf.v.nparams(); }

  size_t FLOPS(int value_or_deriv) const{ return cpt.FLOPS(value_or_deriv) + leaf.v.FLOPS(value_or_deriv); }
  
  //off measured from *end*, return new off
  int getParams(Vector<FloatType> &into, int off) const;

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    cpt.resizeInputBuffer(to);
    leaf.v.resizeInputBuffer(to);
  }

};

#define LAYER_TYPE MatrixTensorContractLayer<FLOATTYPE(U),TensDim,INPUTTYPE(U),DDST(u)>
template<int TensDim, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto matrix_tensor_contract_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<U>(u), weights);
}
#undef LAYER_TYPE

#include "implementation/MatrixTensorContractLayer.tcc"
