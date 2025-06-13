#pragma once
#include "LayerCommon.hpp"
#include <Padding.hpp>

template<typename _FloatType, typename _InputType, typename Store, typename ActivationFunc, typename PaddingFunc>
class ConvolutionLayer1D{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LAYEROUTPUTTYPE(typename Store::type) LayerInputTensorType; //expect a Tensor
  static_assert(LayerInputTensorType::dimension() == 3); //[channel][1d data idx][batch_idx]
  
private:
  Store leaf;
  int _input_tens_size[LayerInputTensorType::dimension()];
  Tensor<FloatType,3> filter; //[depth channel][channel][1d kernel idx]

  ActivationFunc activation_func;
  PaddingFunc padding_func;
  
  int depth;
  int channels;
  int kernel_size;
  int stride;

  bool init;
  int padded_data_len;
  int batch_size;

  mutable FLOPScounter value_FLOPS;
  mutable FLOPScounter deriv_FLOPS;
  
  //Storage from last call to "value"
  //Buffer size > 1 depending on rank if doing pipelining
  mutable RingBuffer<Tensor<FloatType,3> > leaf_buf;
  mutable RingBuffer<Tensor<FloatType,3> > activation_deriv_buf;
  //size_t calls;

  //bool pipeline_mode;

public:
  typedef LeafTag tag;

  //Create a 1Dconvolutional layer. The filter should be of size out_channels * in_channels * kernel_size
  ConvolutionLayer1D(Store &&leaf, const Tensor<FloatType,3> &_filter, 
		     const ActivationFunc &activation_func, const PaddingFunc &padding_func, int stride=1):
    leaf(std::move(leaf)), filter(_filter), init(false), depth(_filter.size(0)), channels(_filter.size(1)), kernel_size(_filter.size(2)),
    activation_func(activation_func), padding_func(padding_func), stride(stride){
  }
  
  ConvolutionLayer1D(const ConvolutionLayer1D &r) = delete;
  ConvolutionLayer1D(ConvolutionLayer1D &&r) = default;
  
  //Forward pass, output  [depth channel][1d data idx][batch_idx]
  Tensor<FloatType,3> value(const InputType &x);

  int deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  int update(int off, const Vector<FloatType> &new_params);
  
  int step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{
    return depth*channels*kernel_size + leaf.v.nparams();
  }

  size_t FLOPS(int value_or_deriv) const{ return (value_or_deriv == 0 ? value_FLOPS.value() : deriv_FLOPS.value()) + leaf.v.FLOPS(value_or_deriv); }
  
  //off measured from *end*, return new off
  int getParams(Vector<FloatType> &into, int off);

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    leaf_buf.resize(to);
    activation_deriv_buf.resize(to);
    leaf.v.resizeInputBuffer(to);
  }

};

template<typename U, typename ActivationFunc, typename PaddingFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto conv1d_layer(U &&u, const Tensor<FLOATTYPE(U),3> &filter, const ActivationFunc &activation_func, const PaddingFunc &padding_func, int stride = 1)
  ->ConvolutionLayer1D<FLOATTYPE(U),INPUTTYPE(U),DDST(u),ActivationFunc,PaddingFunc>{
  return ConvolutionLayer1D<FLOATTYPE(U),INPUTTYPE(U),DDST(u),ActivationFunc,PaddingFunc>(std::forward<U>(u),filter,activation_func,padding_func,stride);
}

#include "implementation/ConvolutionLayer1D.tcc"
