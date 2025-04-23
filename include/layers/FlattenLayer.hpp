#pragma once
#include "LayerCommon.hpp"

//Flatten the input tensor on all dimensions but the last (batch) dimension
template<typename _FloatType, typename _InputType, typename Store>
class FlattenLayer{
public:
  typedef _FloatType FloatType;
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

  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    leaf.v.resizeInputBuffer(to);
  }

};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto flatten_layer(U &&u)->FlattenLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u)>{
  return FlattenLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u)>(std::forward<U>(u));
}

#include "implementation/FlattenLayer.tcc"
