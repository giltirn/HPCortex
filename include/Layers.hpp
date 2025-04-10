#pragma once
#include <Tensors.hpp>
#include <InstanceStorage.hpp>
#include <ActivationFuncs.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>
#include <Padding.hpp>

//Tag for all "leaf" types that can be used to build a model tree
struct LeafTag{};
#define ISLEAF(a) std::is_same<typename std::decay<a>::type::tag,LeafTag>::value
#define FLOATTYPE(a) typename std::decay<a>::type::FloatType
#define INPUTTYPE(a) typename std::decay<a>::type::InputType
#define LAYEROUTPUTTYPE(a) typename std::decay<decltype( std::declval<typename std::decay<a>::type&>().value( std::declval<INPUTTYPE(a)>() ) )>::type

//The input layer
//This is always the lowest layer in the model
template<typename _FloatType, typename _InputType = Matrix<_FloatType> >
class InputLayer{  
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LeafTag tag;
  
  inline InputLayer(){}
  inline InputLayer(InputLayer &&r) = default;
  inline InputLayer(const InputLayer &r) = delete;

  inline const InputType &value(const InputType &x){
    //Simply reflect the passed-down input value back up to commence forwards propagation
    return x;
  }

  //input_above_deriv_return is the derivative of the cost with respect to the inputs
  inline void deriv(Vector<FloatType> &cost_deriv, int off, InputType &&above_deriv, InputType* input_above_deriv_return = nullptr) const{
    //We don't have to do anything for backpropagation as this is the last layer
    if(input_above_deriv_return) *input_above_deriv_return = std::move(above_deriv); //copy back the input derivative if desired
  }
  
  inline void update(int off, const Vector<FloatType> &new_params){}

  inline void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  
  inline int nparams() const{ return 0; }

  inline void getParams(Vector<FloatType> &into, int off){}

  //For pipelining
  inline void resizeInputBuffer(size_t to){}
};

template<typename FloatType, typename InputType = Matrix<FloatType> >
inline InputLayer<FloatType,InputType> input_layer(){ return InputLayer<FloatType,InputType>(); }


//A fully-connected DNN layer
template<typename _FloatType, typename _InputType, typename Store, typename ActivationFunc>
class DNNlayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  Matrix<FloatType> weights;
  Vector<FloatType> bias;  
  Store leaf;
  int size0;
  int size1;

  ActivationFunc activation_func;

  //Storage from last call to "value"
  //Buffer size > 1 depending on rank if doing pipelining
  mutable RingBuffer<Matrix<FloatType> > leaf_buf;
  mutable RingBuffer<Matrix<FloatType> > activation_deriv_buf;
  size_t calls;

  bool pipeline_mode;
  int batch_size;
public:
  typedef LeafTag tag;
  
  DNNlayer(Store &&leaf, const Matrix<FloatType> &weights,const Vector<FloatType> &bias, const ActivationFunc &activation_func):
    leaf(std::move(leaf)), weights(weights), bias(bias),
    size0(weights.size(0)), size1(weights.size(1)),
    activation_func(activation_func), leaf_buf(1), calls(0), pipeline_mode(false), batch_size(0)
  {  }
  DNNlayer(const DNNlayer &r) = delete;
  DNNlayer(DNNlayer &&r) = default;
  
  //Forward pass
  Matrix<FloatType> value(const InputType &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return size0*size1 + size0 + leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    pipeline_mode = true;
    leaf_buf.resize(to);
    activation_deriv_buf.resize(to);
    leaf.v.resizeInputBuffer(to);
  }

};

template<typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights,const Vector<FLOATTYPE(U)> &bias, const ActivationFunc &activation)->DNNlayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u),ActivationFunc>{
  return DNNlayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u),ActivationFunc>(std::forward<U>(u), weights, bias, activation);
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights,const Vector<FLOATTYPE(U)> &bias)->DNNlayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u),noActivation<FLOATTYPE(U)> >{
  return DNNlayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u),noActivation<FLOATTYPE(U)> >(std::forward<U>(u), weights, bias, noActivation<FLOATTYPE(U)>());
}

template<typename _FloatType, typename _InputType, typename ChainInternal, typename ChainBelow>
class skipConnection{
  public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
private:
  ChainBelow leaf_below;
  ChainInternal leaf_internal; //must terminate on an InputLayer (even though it's not really an input layer)
  size_t in_size;
  size_t batch_size;
  mutable RingBuffer<Matrix<FloatType> > in_buf;
public:
  typedef LeafTag tag;
  
  skipConnection(ChainInternal &&leaf_internal, ChainBelow &&leaf_below):
    leaf_below(std::move(leaf_below)), leaf_internal(std::move(leaf_internal)),  in_buf(1), batch_size(0), in_size(0){  }
  skipConnection(const skipConnection &r) = delete;
  skipConnection(skipConnection &&r) = default;
  
  //Forward pass
  Matrix<FloatType> value(const InputType &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;
  
  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{ return leaf_internal.v.nparams() + leaf_below.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    in_buf.resize(to);
    leaf_internal.v.resizeInputBuffer(to);
    leaf_below.v.resizeInputBuffer(to);
  }
};

#define LAYER_TYPE skipConnection<FLOATTYPE(Internal),INPUTTYPE(Below),DDST(internal),DDST(below)>

template<typename Internal, typename Below, typename std::enable_if<ISLEAF(Internal) && ISLEAF(Below), int>::type = 0>
auto skip_connection(Internal &&internal, Below &&below)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<Internal>(internal),std::forward<Below>(below));
}
#undef LAYER_TYPE



//Flatten the input tensor on all dimensions but the last (batch) dimension
template<typename _FloatType, typename _InputType, typename Store>
class FlattenLayer{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LAYEROUTPUTTYPE(typename Store::type) LayerInputTensorType; //expect a Tensor ;; //TODO, add a compile-time test to ensure this!  
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
  
  //Storage from last call to "value"
  //Buffer size > 1 depending on rank if doing pipelining
  mutable RingBuffer<Tensor<FloatType,3> > leaf_buf;
  mutable RingBuffer<Tensor<FloatType,3> > activation_deriv_buf;
  //size_t calls;

  //bool pipeline_mode;

public:
  typedef LeafTag tag;

  ConvolutionLayer1D(Store &&leaf, const Tensor<FloatType,3> &_filter, 
		     const ActivationFunc &activation_func, const PaddingFunc &padding_func, int stride=1):
    leaf(std::move(leaf)), filter(_filter), init(false), depth(_filter.size(0)), channels(_filter.size(1)), kernel_size(_filter.size(2)),
    activation_func(activation_func), padding_func(padding_func), stride(stride){
  }
  
  ConvolutionLayer1D(const ConvolutionLayer1D &r) = delete;
  ConvolutionLayer1D(ConvolutionLayer1D &&r) = default;
  
  //Forward pass, output  [depth channel][1d data idx][batch_idx]
  Tensor<FloatType,3> value(const InputType &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  void update(int off, const Vector<FloatType> &new_params);
  
  void step(int off, const Vector<FloatType> &derivs, FloatType eps);

  //accumulated #params for layers here and below
  inline int nparams() const{
    return depth*channels*kernel_size + leaf.v.nparams();
  }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off);

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

#include "implementation/Layers.tcc"
