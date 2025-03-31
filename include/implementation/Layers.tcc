template<typename FloatType, typename InputType, typename Store, typename ActivationFunc>
Matrix<FloatType> DNNlayer<FloatType,InputType,Store,ActivationFunc>::value(const InputType &x){
  ++calls;
    
  Matrix<FloatType> in = leaf.v.value(x);
  batch_size = in.size(1);
  assert(in.size(0) == size1);
   
  Matrix<FloatType> out = axpyMatThinMat(weights, in, bias);

  //Apply activation function ; modifies output in-place and returns derivatives   
  Matrix<FloatType> activation_deriv;
  activation_func(out, &activation_deriv);
  assert(activation_deriv.size(0) == size0);
  assert(activation_deriv.size(1) == batch_size);

  leaf_buf.push(std::move(in));
  activation_deriv_buf.push(std::move(activation_deriv));
    
  return out;
}

template<typename FloatType, typename InputType, typename Store, typename ActivationFunc>
void DNNlayer<FloatType,InputType,Store,ActivationFunc>::deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return) const{
  profileStart();
  assert(_above_deriv.size(0) == size0);
  assert(_above_deriv.size(1) == batch_size);
  int p=off;
  Matrix<FloatType> layer_deriv;
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

    labelRegionBegin("precompute_activated_derivs");
    Matrix<FloatType> activated_above_deriv = computeThinMatOuterProd(above_deriv,activation_deriv);
    labelRegionEnd();
      
    //Compute layer deriv
    labelRegionBegin("compute_layer_deriv");
    layer_deriv = mulMatTransposeThinMat(weights, activated_above_deriv);
    labelRegionEnd();

    //Now we finish up the derivs wrt our parameters      
    //dcost / dw_jk = \sum_i (dcost/df_i df_i/dg_i) dg_i/dw_jk
    //dcost / db_j = \sum_i (dcost/df_i df_i/dg_i) dg_i/db_j
      
    //dg_i / d w_jk = delta_ij x_k
    //dg_i / d b_j = delta_ij
      
    {
      labelRegionBegin("cost_deriv_weights");
      size_t bs = batch_size;
	
      //this version saves the overheads of the intermediate copy
      autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
      thinMulMatMatTranspose_p(cost_deriv_v.data() + p, activated_above_deriv, in);     
	
      p += size0*size1;
      labelRegionEnd();
      labelRegionBegin("cost_deriv_bias");
      autoView(activated_above_deriv_v,activated_above_deriv,DeviceRead);
      accelerator_for2d(dummy1,1, j,size0, 64,{
	  int pp = p + j;
	  FloatType v = activated_above_deriv_v(j,0);
	  for(int b=1;b<bs;b++)
	    v += activated_above_deriv_v(j,b);
	  cost_deriv_v(pp) = v;
	});

      p += size0;

      labelRegionEnd();
    }
    
  }//close views and free temporaries before calling layer below
    
  leaf.v.deriv(cost_deriv, p, std::move(layer_deriv), input_above_deriv_return);
  profileStop();
}

template<typename FloatType, typename InputType, typename Store, typename ActivationFunc>
void DNNlayer<FloatType,InputType,Store,ActivationFunc>::update(int off, const Vector<FloatType> &new_params){
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

template<typename FloatType, typename InputType, typename Store, typename ActivationFunc>
void DNNlayer<FloatType,InputType,Store,ActivationFunc>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
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



  //off measured from *end*, return new off
template<typename FloatType, typename InputType, typename Store, typename ActivationFunc>
void DNNlayer<FloatType,InputType,Store,ActivationFunc>::getParams(Vector<FloatType> &into, int off){
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



template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
Matrix<FloatType> skipConnection<FloatType,InputType,ChainInternal,ChainBelow>::value(const InputType &x){
  Matrix<FloatType> in = leaf_below.v.value(x);
  Matrix<FloatType> out = in + leaf_internal.v.value(in);
  
  in_buf.push(std::move(in));
  in_size = in.size(0);
  batch_size = in.size(1);
  
  return out;
}

template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
void skipConnection<FloatType,InputType,ChainInternal,ChainBelow>::deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return) const{
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
template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
void skipConnection<FloatType,InputType,ChainInternal,ChainBelow>::update(int off, const Vector<FloatType> &new_params){
  int p=off;
  leaf_internal.v.update(p, new_params);
  p += leaf_internal.v.nparams();
  leaf_below.v.update(p, new_params);
}

template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
void skipConnection<FloatType,InputType,ChainInternal,ChainBelow>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  int p=off;
  leaf_internal.v.step(p, derivs, eps);
  p += leaf_internal.v.nparams();
  leaf_below.v.step(p, derivs, eps);
}

//off measured from *end*, return new off
template<typename FloatType, typename InputType, typename ChainInternal, typename ChainBelow>
void skipConnection<FloatType,InputType,ChainInternal,ChainBelow>::getParams(Vector<FloatType> &into, int off){
  int p = off;
  leaf_internal.v.getParams(into, p);
  p += leaf_internal.v.nparams();
  leaf_below.v.getParams(into,p);
}



template<typename FloatType, typename InputType, typename Store>
Matrix<FloatType> FlattenLayer<FloatType,InputType,Store>::value(const InputType &x){
  LayerInputTensorType in = leaf.v.value(x);
  if(!init){
    memcpy(_input_tens_size, in.sizeArray(), in.dimension() * sizeof(int));
    init = true;
  }
  
  constexpr int tens_dim = in.dimension();
  int batch_size = in.size(tens_dim-1);
  int out_size = 1;
  for(int i=0;i<tens_dim-1;i++)
    out_size *= in.size(i);

  Matrix<FloatType> out(out_size,batch_size);
  autoView(out_v,out,DeviceWrite);
  autoView(in_v,in,DeviceRead);
  accelerator_for2d(b,batch_size, i,out_size, 1,{
      //rely on the fact that the batch index is the fastest moving,  eg. for a 3 tensor   off = b + batch_size*(z + zsize*(y + ysize*x))      i=(z + zsize*(y + ysize*x)) 
      out_v(i,b) = in_v.data()[b + i*batch_size];
    });
  return out;
}
template<typename FloatType, typename InputType, typename Store>
void FlattenLayer<FloatType,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return) const{
  LayerInputTensorType above_deriv_passdown(_input_tens_size);
  {
    Matrix<FloatType> above_deriv_in(std::move(_above_deriv)); //dcost/dvalue_i
    int batch_size = above_deriv_in.size(1);
    size_t flat_size = above_deriv_in.size(0);

    autoView(above_deriv_passdown_v,above_deriv_passdown,DeviceWrite);
    autoView(above_deriv_in_v,above_deriv_in,DeviceRead);
    accelerator_for2d(b,batch_size, i,flat_size, 1,{
	//rely on the fact that the batch index is the fastest moving
	above_deriv_passdown_v.data()[b + i*batch_size] = above_deriv_in_v(i,b);
      });
  }
  leaf.v.deriv(cost_deriv,off, std::move(above_deriv_passdown),input_above_deriv_return);
}
template<typename FloatType, typename InputType, typename Store>
void FlattenLayer<FloatType,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  leaf.v.update(off,new_params);
}
template<typename FloatType, typename InputType, typename Store>   
void FlattenLayer<FloatType,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  leaf.v.step(off,derivs,eps);
}
template<typename FloatType, typename InputType, typename Store>  
void FlattenLayer<FloatType,InputType,Store>::getParams(Vector<FloatType> &into, int off){
  leaf.v.getParams(into,off);
}
