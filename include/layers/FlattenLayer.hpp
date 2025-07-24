#pragma once
#include "LayerCommon.hpp"

//Flatten the input tensor on all dimensions but the last (batch) dimension
template<typename Config, typename _InputType, typename Store>
class FlattenLayer{
public:
  EXTRACT_CONFIG_TYPES;
  typedef _InputType InputType;
  typedef LAYEROUTPUTTYPE(typename Store::type) LayerInputTensorType; //expect a Tensor
  static_assert(std::is_same<LayerInputTensorType, Tensor<FloatType, LayerInputTensorType::dimension()> >::value == true );
private:
  Store leaf;
  int _input_tens_size[LayerInputTensorType::dimension()];
  bool init;
public:
  typedef LeafTag tag;

  FlattenLayer(Store &&leaf): leaf(std::move(leaf)), init(false){}
  
  FlattenLayer(const FlattenLayer &r) = delete;
  FlattenLayer(FlattenLayer &&r) = default;
  
  //Forward pass
  Matrix<FloatType> value(const InputType &x);

  int deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

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

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto flatten_layer(U &&u)->FlattenLayer<CONFIGTYPE(U),INPUTTYPE(U),DDST(u)>{
  return FlattenLayer<CONFIGTYPE(U),INPUTTYPE(U),DDST(u)>(std::forward<U>(u));
}

#include "implementation/FlattenLayer.tcc"
