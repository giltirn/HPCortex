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
  Matrix<FloatType> value(const Matrix<FloatType> &x){
    ++calls;
    
    Matrix<FloatType> in = leaf.v.value(x);
    batch_size = x.size(1);   
    assert(in.size(0) == size1);
    assert(in.size(1) == batch_size);

    leaf_buf.push(in);
    
    Matrix<FloatType> out(size0,batch_size);
    {
      autoView(bias_v,bias,DeviceRead);
      autoView(out_v,out,DeviceWrite);
      autoView(in_v,in,DeviceRead);
      autoView(weights_v,weights,DeviceRead);

      //Basic version where columns are summed over within a thread and rows/batches distributed over threads
      size_t sz1 = size1;
      accelerator_for2d(b,batch_size,i,size0,1,{
	  out_v(i,b) = bias_v(i);
	  for(int j=0;j<sz1;j++)
	    out_v(i,b) += weights_v(i,j)* in_v(j,b);
	});      
    }
	
    Matrix<FloatType> activation = activation_func(out);
    assert(activation.size(0) == size0);
    assert(activation.size(1) == batch_size);

    {
      autoView(out_v,out,DeviceReadWrite);
      autoView(activation_v,activation,DeviceRead);

      accelerator_for2d(b,batch_size,i,size0,1,{
	  out_v(i,b) *= activation_v(i,b);
	});
    }
      
    activation_buf.push(activation); //TODO: make this a move
    
    return out;
  }

  //TODO: Find a way to free 'above_deriv' once used as it is not needed for layers below
  void deriv(Vector<FloatType> &cost_deriv, int off, const Matrix<FloatType> &above_deriv, Matrix<FloatType>* input_above_deriv_copyback = nullptr) const{
    assert(above_deriv.size(0) == size0);
    int p=off;
    Matrix<FloatType> layer_deriv(size1,batch_size,0.);
    {
      //until the pipeline is "primed", the ring buffers will pop uninitialized values. We could in principle skip doing any computation until then
      //but for now we just initialize with zero values (TODO: revisit)
      Matrix<FloatType> in = leaf_buf.isFilled() ? leaf_buf.pop(): Matrix<FloatType>(size1,batch_size,0.);
      assert(in.size(0) == size1);
      assert(in.size(1) == batch_size);
      
      Matrix<FloatType> activation = activation_buf.isFilled() ? activation_buf.pop() : Matrix<FloatType>(size0,batch_size,0.);
      assert(activation.size(0) == size0);
      assert(activation.size(1) == batch_size);      
      
      //for reverse differentiation, we pass down the derivatives with respect to our inputs
      //f(x)_i = act_i b_i + \sum_j act_i w_ij x_j
      //dcost / dx_j = \sum_i dcost/df_i df_i/dx_j
      //df_i/dx_j = act_i w_ij    

      autoView(above_deriv_v,above_deriv,DeviceRead);
      autoView(activation_v,activation,DeviceRead);
    
      {
	autoView(layer_deriv_v,layer_deriv,DeviceReadWrite);
	autoView(weights_v,weights,DeviceRead);

	//Basic implementation
	size_t sz0 = size0;
	accelerator_for2d(b,batch_size,j,size1,1,{
	  for(int i=0;i<sz0;i++)
	    layer_deriv_v(j,b) += above_deriv_v(i,b) * activation_v(i,b) * weights_v(i,j);
	  });
      }

      //Now we finish up the derivs wrt our parameters
      //df(x)_i / d w_jk = delta_ij act_j x_k
      //df(x)_i / d b_j = delta_ij act_j
      //dcost / dw_jk = \sum_i dcost/df_i df_i/dw_jk = dcost/df_j * act_j * x_k
      //dcost / db_j = \sum_i dcost/df_i df_i/db_j = dcost/df_j * act_j
      {
	autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
	autoView(in_v,in,DeviceRead);
	size_t bs = batch_size;
	size_t sz1 = size1;
	accelerator_for2d(k,size1,j,size0,1,{
	    int pp = p + k + sz1*j;	    
	    for(int b=0;b<bs;b++)
	      cost_deriv_v(pp) += above_deriv_v(j,b) * activation_v(j,b) * in_v(k,b); //batch reduction! (assume zero-initialized)
	  });
	p += size0*size1;

	//TODO: Fuse these
	accelerator_for(j,size0,{
	    int pp = p + j;
	    for(int b=0;b<bs;b++)
	      cost_deriv_v(pp) += above_deriv_v(j,b) * activation_v(j,b);
	  });
	p += size0;
      }
    
    }//close views and free temporaries before calling layer below
    
    leaf.v.deriv(cost_deriv, p, layer_deriv, input_above_deriv_copyback);
  }

  void update(int off, const Vector<FloatType> &new_params){
    int p=off;
    {
      autoView(new_params_v,new_params,DeviceRead);
      autoView(bias_v,bias,DeviceWrite);
      autoView(weights_v,weights,DeviceWrite);
      size_t sz1=size1;
      accelerator_for2d(j,size1,i,size0,1,{
	  int pp = p + j + sz1*i;
	  weights_v(i,j) = new_params_v(pp);
	});
	  
      p += size0*size1;

      accelerator_for(i,size0,{
	bias_v(i) = new_params_v(p + i);
	});
      
      p += size0;
    }
    leaf.v.update(p, new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    int p=off;
    {
      autoView(derivs_v,derivs,DeviceRead);
      autoView(bias_v,bias,DeviceReadWrite);
      autoView(weights_v,weights,DeviceReadWrite);
      size_t sz1 = size1;
      accelerator_for2d(j,size1,i,size0,1,{
	  int pp = p + j + sz1*i;
	  weights_v(i,j) -= derivs_v(pp) * eps;
	});
	  
      p += size0*size1;

      accelerator_for(i,size0,{
	bias_v(i) -= derivs_v(p + i) * eps;
	});
      
      p += size0;
    }
    leaf.v.step(p, derivs, eps);
  }

  //accumulated #params for layers here and below
  inline int nparams() const{ return size0*size1 + size0 + leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off){
    int p = off;
    {
      autoView(into_v,into,DeviceReadWrite);
      autoView(bias_v,bias,DeviceRead);
      autoView(weights_v,weights,DeviceRead);
      size_t sz1 = size1;
      accelerator_for2d(j,size1,i,size0,1,{
	  int pp = p + j + sz1*i;
	  into_v(pp) = weights_v(i,j);
	});

      p += size0*size1;
	  
      accelerator_for(i,size0,{
	into_v(p + i) = bias_v(i);
	});

      p += size0;
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

