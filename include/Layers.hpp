#pragma once
#include <Tensors.hpp>
#include <InstanceStorage.hpp>
#include <ActivationFuncs.hpp>
#include <RingBuffer.hpp>

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

  inline void deriv(Vector<FloatType> &cost_deriv, int off, const Matrix<FloatType> &above_deriv, Matrix<FloatType>* input_above_deriv_copyback = nullptr) const{
    //We don't have to do anything for backpropagation as this is the last layer
    if(input_above_deriv_copyback) *input_above_deriv_copyback = above_deriv; //copy back the input derivative if desired
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
  RingBuffer<Matrix<FloatType> > leaf_buf;
  RingBuffer<Matrix<FloatType> > activation_buf;
  size_t calls;

  bool pipeline_mode;
public:
  typedef LeafTag tag;
  
  DNNlayer(Store &&leaf, const Matrix<FloatType> &weights,const Vector<FloatType> &bias, const ActivationFunc &activation_func):
    leaf(std::move(leaf)), weights(weights), bias(bias),
    size0(weights.size(0)), size1(weights.size(1)),
    activation_func(activation_func), leaf_buf(1), calls(0), pipeline_mode(false)
  {  }
  DNNlayer(const DNNlayer &r) = delete;
  DNNlayer(DNNlayer &&r) = default;
  
  //Forward pass
  Matrix<FloatType> value(const Matrix<FloatType> &x){
    ++calls;
    
    Matrix<FloatType> in = leaf.v.value(x);
    int batch_size = x.size(1);   
    assert(in.size(0) == size1);
    assert(in.size(1) == batch_size);

    leaf_buf.push(in);
    //if(pipeline_mode) std::cout << "RANK " << rank << " " << this << " CALL " << calls << " INPUT " << x << " VALUE BUFFERED INPUT " << in << std::endl;
    //else std::cout << "RANK " << rank << " " << this << " UNPIPELINED CALL " << calls << " INPUT " << x << " VALUE BUFFERED INPUT " << in << std::endl;
    
    Matrix<FloatType> out(size0,batch_size,0.0);
    autoView(bias_v,bias,HostRead);
    
    for(int i=0;i<size0;i++){
      for(int b=0;b<batch_size;b++){
	out(i,b) = bias_v(i);
	for(int j=0;j<size1;j++)
	  out(i,b) += weights(i,j)* in(j,b);
      }
    }
	
    Matrix<FloatType> activation = activation_func(out);
    assert(activation.size(0) == size0);
    assert(activation.size(1) == batch_size);
    
    for(int i=0;i<size0;i++)
      for(int b=0;b<batch_size;b++)
	out(i,b) *= activation(i,b);    

    activation_buf.push(activation);
    
    return out;
  }
 
  void deriv(Vector<FloatType> &cost_deriv, int off, const Matrix<FloatType> &above_deriv, Matrix<FloatType>* input_above_deriv_copyback = nullptr) const{
    assert(above_deriv.size(0) == size0);
    Matrix<FloatType> in = leaf_buf.pop();
    Matrix<FloatType> activation = activation_buf.pop();
    int batch_size = in.size(1);

    //if(pipeline_mode) std::cout << "RANK " << rank << " " << this << " CALL " << calls << " DERIV USING BUFFERED INPUT " << in << " ABOVE_DERIV " << above_deriv << " WITH INPUT COST DERIV " << cost_deriv;
    //else std::cout << "RANK " << rank << " " << this << " UNPIPELINED CALL " << calls << " DERIV USING BUFFERED INPUT " << in << " ABOVE_DERIV " << above_deriv << " WITH INPUT COST DERIV " << cost_deriv;
    
    //for reverse differentiation, we pass down the derivatives with respect to our inputs
    //f(x)_i = act_i b_i + \sum_j act_i w_ij x_j
    //dcost / dx_j = \sum_i dcost/df_i df_i/dx_j
    //df_i/dx_j = act_i w_ij
    Matrix<FloatType> layer_deriv(size1,batch_size,0.);
    for(int j=0;j<size1;j++)
      for(int i=0;i<size0;i++)
	for(int b=0;b<batch_size;b++)
	  layer_deriv(j,b) += above_deriv(i,b) * activation(i,b) * weights(i,j);

    //Now we finish up the derivs wrt our parameters
    //df(x)_i / d w_jk = delta_ij act_j x_k
    //df(x)_i / d b_j = delta_ij act_j
    //dcost / dw_jk = \sum_i dcost/df_i df_i/dw_jk = dcost/df_j * act_j * x_k
    //dcost / db_j = \sum_i dcost/df_i df_i/db_j = dcost/df_j * act_j
    int p=off;
    {
      autoView(cost_deriv_v,cost_deriv,HostReadWrite);
      for(int j=0;j<size0;j++)
	for(int k=0;k<size1;k++){
	  for(int b=0;b<batch_size;b++)
	    cost_deriv_v(p) += above_deriv(j,b) * activation(j,b) * in(k,b); //batch reduction! (assume zero-initialized)
	  ++p;
	}
	
      for(int j=0;j<size0;j++){
	for(int b=0;b<batch_size;b++)
	  cost_deriv_v(p) += above_deriv(j,b) * activation(j,b);
	++p;
      }
    }
    
    //std::cout << " AND RESULT " << cost_deriv << std::endl;
    
    leaf.v.deriv(cost_deriv, p, layer_deriv, input_above_deriv_copyback);
  }

  void update(int off, const Vector<FloatType> &new_params){
    int p=off;
    {
      autoView(new_params_v,new_params,HostRead);
      autoView(bias_v,bias,HostWrite);
      for(int i=0;i<size0;i++)
	for(int j=0;j<size1;j++)
	  weights(i,j) = new_params_v(p++);
      for(int i=0;i<size0;i++)
	bias_v(i) = new_params_v(p++);
    }
    leaf.v.update(p, new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    int p=off;
    {
      autoView(derivs_v,derivs,HostRead);
      autoView(bias_v,bias,HostReadWrite);
      
      for(int i=0;i<size0;i++)
	for(int j=0;j<size1;j++){
	  //std::cout << "Weights " << i << " " << j << " " << weights(i,j) << " -= " << derivs(p) << "*" << eps;
	  weights(i,j) -= derivs_v(p++) * eps;
	  //std::cout << " = " <<  weights(i,j) << std::endl;
	}
      for(int i=0;i<size0;i++){
	//std::cout << "Bias " << i << " " << bias(i) << " -= " << derivs(p) << "*" << eps;
	bias_v(i) -= derivs_v(p++) * eps;
	//std::cout << " = " << bias(i) << std::endl;
      }
    }
    leaf.v.step(p, derivs, eps);
  }

  //accumulated #params for layers here and below
  inline int nparams() const{ return size0*size1 + size0 + leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off){
    int p = off;
    {
      autoView(into_v,into,HostReadWrite);
      autoView(bias_v,bias,HostRead);
      for(int i=0;i<size0;i++)
	for(int j=0;j<size1;j++)
	  into_v(p++) = weights(i,j);
      for(int i=0;i<size0;i++)
	into_v(p++) = bias_v(i);
    }
    leaf.v.getParams(into, p);
  }

  //For pipelining
  inline void resizeInputBuffer(size_t to){
    //std::cout << "RANK " << rank << " " << this << " RESIZING RING BUFFERS TO " << to << std::endl;
    pipeline_mode = true;
    leaf_buf.resize(to);
    activation_buf.resize(to);
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

