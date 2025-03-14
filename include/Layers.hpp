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
  Matrix<FloatType> value(const Matrix<FloatType> &x){
    ++calls;
    
    Matrix<FloatType> in = leaf.v.value(x);
    batch_size = x.size(1);   
    assert(in.size(0) == size1);
    assert(in.size(1) == batch_size);
   
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

    //Apply activation function ; modifies output in-place and returns derivatives   
    Matrix<FloatType> activation_deriv;
    activation_func(out, &activation_deriv);
    assert(activation_deriv.size(0) == size0);
    assert(activation_deriv.size(1) == batch_size);

    leaf_buf.push(std::move(in));
    activation_deriv_buf.push(std::move(activation_deriv));
    
    return out;
  }

  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, Matrix<FloatType>* input_above_deriv_return = nullptr) const{
    assert(_above_deriv.size(0) == size0);
    assert(_above_deriv.size(1) == batch_size);
    int p=off;
    Matrix<FloatType> layer_deriv(size1,batch_size,0.);
    {
      Matrix<FloatType> above_deriv(std::move(_above_deriv)); //inside the braces above ensures this object is freed before the next layer is called
      
      //until the pipeline is "primed", the ring buffers will pop uninitialized values. We could in principle skip doing any computation until then
      //but for now we just initialize with zero values (TODO: revisit)
      Matrix<FloatType> in = leaf_buf.isFilled() ? leaf_buf.pop(): Matrix<FloatType>(size1,batch_size,0.);
      assert(in.size(0) == size1);
      assert(in.size(1) == batch_size);
      
      Matrix<FloatType> activation_deriv = activation_deriv_buf.isFilled() ? activation_deriv_buf.pop() : Matrix<FloatType>(size0,batch_size,0.);
      assert(activation_deriv.size(0) == size0);
      assert(activation_deriv.size(1) == batch_size);      

      //for reverse differentiation, we pass down the derivatives of the cost with respect to our inputs, x (vector)
      //Write output  f_i(x) = A( g_i(x) )   where A is the activation function
      //where g_i(x) = b_i + \sum_j w_ij x_j
      //
      //dcost / dx_j = \sum_i dcost/df_i df_i/dx_j
      //df_i/dx_j = df_i / dg_i dg_i / dx_j
      //dg_i/dx_j = w_ij
      
      //dcost / dx_j = \sum_i (dcost/df_i df_i/dg_i) w_ij     :  "layer deriv"
      
      //precompute the "activated derivatives"  (dcost/df_i df_i/dg_i) as they are reused below
      Matrix<FloatType> activated_above_deriv(size0,batch_size);
      {
	autoView(above_deriv_v,above_deriv,DeviceRead);
	autoView(activation_deriv_v,activation_deriv,DeviceRead);
	autoView(activated_above_deriv_v,activated_above_deriv,DeviceWrite);

	accelerator_for2d(b,batch_size,i,size0,1,{
	    activated_above_deriv_v(i,b) = above_deriv_v(i,b) * activation_deriv_v(i,b);
	  });
      }

      //Compute layer deriv
      autoView(activated_above_deriv_v,activated_above_deriv,DeviceRead);
    
      {
	autoView(layer_deriv_v,layer_deriv,DeviceReadWrite);
	autoView(weights_v,weights,DeviceRead);

	//Basic implementation
	size_t sz0 = size0;
	accelerator_for2d(b,batch_size,j,size1,1,{
	  for(int i=0;i<sz0;i++)
	    layer_deriv_v(j,b) += activated_above_deriv_v(i,b) * weights_v(i,j);
	  });
      }

      //Now we finish up the derivs wrt our parameters      
      //dcost / dw_jk = \sum_i (dcost/df_i df_i/dg_i) dg_i/dw_jk
      //dcost / db_j = \sum_i (dcost/df_i df_i/dg_i) dg_i/db_j
      
      //dg_i / d w_jk = delta_ij x_k
      //dg_i / d b_j = delta_ij
      
      {
	autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
	autoView(in_v,in,DeviceRead);
	size_t bs = batch_size;
	size_t sz1 = size1;
	accelerator_for2d(k,size1,j,size0,1,{
	    int pp = p + k + sz1*j;	    
	    for(int b=0;b<bs;b++)
	      cost_deriv_v(pp) += activated_above_deriv_v(j,b) * in_v(k,b); //batch reduction! (assume zero-initialized)
	  });
	p += size0*size1;

	//TODO: Fuse these
	accelerator_for(j,size0,{
	    int pp = p + j;
	    for(int b=0;b<bs;b++)
	      cost_deriv_v(pp) += activated_above_deriv_v(j,b);
	  });
	p += size0;
      }
    
    }//close views and free temporaries before calling layer below
    
    leaf.v.deriv(cost_deriv, p, std::move(layer_deriv), input_above_deriv_return);
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
  Matrix<FloatType> value(const Matrix<FloatType> &x){
    Matrix<FloatType> in = leaf_below.v.value(x);
    Matrix<FloatType> out = in + leaf_internal.v.value(in);
    
    in_buf.push(std::move(in));
    in_size = in.size(0);
    batch_size = in.size(1);

    return out;
  }

  void deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, Matrix<FloatType>* input_above_deriv_return = nullptr) const{
    assert(_above_deriv.size(0) == in_size);
    assert(_above_deriv.size(1) == batch_size);
    int p=off;
    Matrix<FloatType> layer_deriv;
    {
      Matrix<FloatType> above_deriv(std::move(_above_deriv)); //inside the braces above ensures this object is freed before the next layer is called
      
      //until the pipeline is "primed", the ring buffers will pop uninitialized values. We could in principle skip doing any computation until then
      //but for now we just initialize with zero values (TODO: revisit)
      Matrix<FloatType> in = in_buf.isFilled() ? in_buf.pop(): Matrix<FloatType>(in_size,batch_size,0.);
      assert(in.size(0) == in_size);
      assert(in.size(1) == batch_size);
      
      //f_i(x) = g_i(x) + x_i

      //deriv wrt inputs for backprop
      //df_i/dx_j = dg_i/dx_j + delta_ij
      
      //dcost / dx_j = \sum_i dcost/df_i df_i/dx_j
      //             = \sum_i dcost/df_i dg_i/dx_j  + \sum_i dcost/df_i delta_ij
      //             = \sum_i dcost/df_i dg_i/dx_j  + \sum_i dcost/df_j
      
      //deriv wrt params for filling cost_deriv
      //df_i/dparam_p = dg_i/dparam_p

      layer_deriv = above_deriv; //dcost/df_j
      Matrix<FloatType> leaf_internal_deriv; //\sum_i dcost/df_i dg_i/dx_j
      leaf_internal.v.deriv(cost_deriv, p, std::move(above_deriv), &leaf_internal_deriv);

      layer_deriv += leaf_internal_deriv;

      p += leaf_internal.v.nparams();  
    }//close views and free temporaries before calling layer below
    
    leaf_below.v.deriv(cost_deriv, p, std::move(layer_deriv), input_above_deriv_return);
  }

  void update(int off, const Vector<FloatType> &new_params){
    int p=off;
    leaf_internal.v.update(p, new_params);
    p += leaf_internal.v.nparams();
    leaf_below.v.update(p, new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    int p=off;
    leaf_internal.v.step(p, derivs, eps);
    p += leaf_internal.v.nparams();
    leaf_below.v.step(p, derivs, eps);
  }


  //accumulated #params for layers here and below
  inline int nparams() const{ return leaf_internal.v.nparams() + leaf_below.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector<FloatType> &into, int off){
    int p = off;
    leaf_internal.v.getParams(into, p);
    p += leaf_internal.v.nparams();
    leaf_below.v.getParams(into,p);
  }

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
