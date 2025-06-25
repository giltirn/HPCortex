#pragma once
#include "LayerCommon.hpp"

//Unflatten an input matrix into a tensor, lexicographically; performs the inverse of FlattenLayer
template<typename _FloatType, int OutDimension, typename _InputType, typename Store>
class UnflattenLayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LAYEROUTPUTTYPE(typename Store::type) LayerInputTensorType; //expect a Matrix
  static_assert(std::is_same<LayerInputTensorType, Matrix<FloatType> >::value == true );
private:
  Store leaf;
  int _output_tens_size[OutDimension];
public:
  typedef LeafTag tag;

  UnflattenLayer(Store &&leaf, int const* output_tens_size): leaf(std::move(leaf)){
    memcpy(_output_tens_size, output_tens_size, OutDimension*sizeof(int));
  }
  
  UnflattenLayer(const UnflattenLayer &r) = delete;
  UnflattenLayer(UnflattenLayer &&r) = default;
  
  //Forward pass
  Tensor<FloatType, OutDimension> value(const InputType &x);

  int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType, OutDimension> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  int update(int off, const Vector<FloatType> &new_params);
  
  int step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return leaf.v.nparams(); }

  size_t FLOPS(int value_or_deriv) const{ return leaf.v.FLOPS(value_or_deriv); }
  
  //off measured from *end*, return new off
  int getParams(Vector<FloatType> &into, int off) const;

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    leaf.v.resizeInputBuffer(to);
  }

};

template<int OutDimension, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto unflatten_layer(int const* output_tens_dim, U &&u)->UnflattenLayer<FLOATTYPE(U),OutDimension,INPUTTYPE(U),DDST(u)>{
  return UnflattenLayer<FLOATTYPE(U),OutDimension,INPUTTYPE(U),DDST(u)>(std::forward<U>(u), output_tens_dim);
}

#include "implementation/UnflattenLayer.tcc"
