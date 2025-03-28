#pragma once
#include <Tensors.hpp>
#include <InstanceStorage.hpp>
#include <ActivationFuncs.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//Tag for all "leaf" types that can be used to build a model tree
struct LeafTag{};
#define ISLEAF(a) std::is_same<typename std::decay<a>::type::tag,LeafTag>::value
#define FLOATTYPE(a) typename std::decay<a>::type::FloatType

//The input layer
//This is always the lowest layer in the model
template<typename _FloatType>
class InputLayer{  
public:
  typedef _FloatType FloatType;
  typedef LeafTag tag;
  
  inline InputLayer(){}
  inline InputLayer(InputLayer &&r) = default;
  inline InputLayer(const InputLayer &r) = delete;
  
  inline const Matrix<FloatType> &value(const Matrix<FloatType> &x){
    //Simply reflect the passed-down input value back up to commence forwards propagation
    return x;
  }

  inline void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&above_deriv, Matrix<FloatType>* input_above_deriv_return = nullptr) const{
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

template<typename FloatType>
inline InputLayer<FloatType> input_layer(){ return InputLayer<FloatType>(); }


//A fully-connected DNN layer
template<typename _FloatType, typename Store, typename ActivationFunc>
class DNNlayer{
public:
  typedef _FloatType FloatType;
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
  Matrix<FloatType> value(const Matrix<FloatType> &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, Matrix<FloatType>* input_above_deriv_return = nullptr) const;

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
auto dnn_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights,const Vector<FLOATTYPE(U)> &bias, const ActivationFunc &activation)->DNNlayer<FLOATTYPE(U),DDST(u),ActivationFunc>{
  return DNNlayer<FLOATTYPE(U),DDST(u),ActivationFunc>(std::forward<U>(u), weights, bias, activation);
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix<FLOATTYPE(U)> &weights,const Vector<FLOATTYPE(U)> &bias)->DNNlayer<FLOATTYPE(U),DDST(u),noActivation<FLOATTYPE(U)> >{
  return DNNlayer<FLOATTYPE(U),DDST(u),noActivation<FLOATTYPE(U)> >(std::forward<U>(u), weights, bias, noActivation<FLOATTYPE(U)>());
}

template<typename _FloatType, typename ChainInternal, typename ChainBelow>
class skipConnection{
  public:
  typedef _FloatType FloatType;
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
  Matrix<FloatType> value(const Matrix<FloatType> &x);

  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, Matrix<FloatType>* input_above_deriv_return = nullptr) const;
  
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

#define LAYER_TYPE skipConnection<FLOATTYPE(Internal),DDST(internal),DDST(below)>

template<typename Internal, typename Below, typename std::enable_if<ISLEAF(Internal) && ISLEAF(Below), int>::type = 0>
auto skip_connection(Internal &&internal, Below &&below)-> LAYER_TYPE{
  return LAYER_TYPE(std::forward<Internal>(internal),std::forward<Below>(below));
}
#undef LAYER_TYPE


#include "implementation/Layers.tcc"
