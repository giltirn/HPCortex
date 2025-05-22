template<typename FloatType, typename InputType, typename Store, typename ActivationFunc, typename PaddingFunc>
Tensor<FloatType,3> ConvolutionLayer1D<FloatType,InputType,Store,ActivationFunc,PaddingFunc>::value(const InputType &x){
  LayerInputTensorType in = leaf.v.value(x);

  //std::cout << "Conv1d input tensor of size " << in.sizeArrayString() << std::endl;
  
  in = padding_func.padInput(in);

  //std::cout << "Conv1d padded input to size " << in.sizeArrayString() << std::endl;
  
  if(!init){
    batch_size = in.size(2);
    padded_data_len = in.size(1);
    init=true;
  }else{
    assert(in.size(2) == batch_size && in.size(1) == padded_data_len);
  }
  assert(in.size(0) == channels);
  assert(padded_data_len >= kernel_size);

  int out_data_len = (padded_data_len - kernel_size + stride)/stride;

  //std::cout << "Conv1d output data length ( padded_data_len="  << padded_data_len << " -  kernel_size=" << kernel_size << " + stride=" << stride << ")/stride gives " << out_data_len << std::endl;
  
  //A convolution is just a multi-dim generalization of a dot product
  int out_size[3] = { depth, out_data_len, batch_size };
  Tensor<FloatType, 3> out(out_size, 0.);

  //std::cout << "Conv1d output size " << out.sizeArrayString() << std::endl;
  
  {
    autoView(out_v,out,DeviceReadWrite);
    autoView(in_v,in,DeviceRead);
    autoView(filter_v,filter,DeviceRead); //[depth channel][channel][1d kernel idx]
    int kernel_size_ = kernel_size;
    int stride_ = stride;
    
    //Loop over channels for a given output depth channel, summing into output
    for(int d=0;d<depth;d++){     //TODO: accelerator_for4d?
      accelerator_for3d(b,batch_size, o, out_data_len, c, channels,  1, { 
	  FloatType *fdc = &filter_v(d,c,0); //TODO: place in shm
	  FloatType tmp = 0.;
	  for(int k=0;k<kernel_size_;k++)
	    tmp += fdc[k] * in_v(c,o*stride_+k,b);
	  atomicAdd(&out_v(d,o,b), tmp);
	});
    }
  }   
			   
  //Apply activation function ; modifies output in-place and returns derivatives   
  Tensor<FloatType,3> activation_deriv;
  activation_func(out, &activation_deriv);
  assert(activation_deriv.size(0) == depth);
  assert(activation_deriv.size(1) == out_data_len);
  assert(activation_deriv.size(2) == batch_size);

  leaf_buf.push(std::move(in)); //keep the *padded* tensor
  activation_deriv_buf.push(std::move(activation_deriv));
    
  return out;
}

template<typename FloatType, typename InputType, typename Store, typename ActivationFunc, typename PaddingFunc>
int ConvolutionLayer1D<FloatType,InputType,Store,ActivationFunc,PaddingFunc>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return) const{
  assert(init);

  //Out channel index = d
  //Out data index = o
  //In channel index c
  //Index in filter k = 0..K-1
  //Stride s
  
  //Write output  f_do(x) = A( g_do(x) )   where A is the activation function and x is the *padded* data
  //g_do(x) = \sum_c [ \sum_k filter_{dck} x_{c,s*o+k} ]
  //
  //dcost / dx_ci = \sum_do dcost/df_do df_do/dx_ci
  //df_do/dx_ci = df_do / dg_do dg_do / dx_ci
  //dg_do/dx_ci = \sum_k filter_{dck} delta_{s*o+k,i}


  //dcost / dx_ci = \sum_do (dcost/df_do df_do / dg_do) [ \sum_k filter_{dck} delta_{s*o+k,i} ]  :  "layer deriv"
  //dcost/df_do : "above_deriv"
  //(dcost/df_do df_do / dg_do) : "activated derivatives"
  
  int p=off;
  int in_sz[3] = {channels,padded_data_len,batch_size};
  Tensor<FloatType,3> layer_deriv(in_sz);
  {
    Tensor<FloatType,3> above_deriv(std::move(_above_deriv)); //inside the braces above ensures this object is freed before the next layer is called
      
    //until the pipeline is "primed", the ring buffers will pop uninitialized values. We could in principle skip doing any computation until then
    //but for now we just initialize with zero values (TODO: revisit)

    Tensor<FloatType,3> in = leaf_buf.isFilled() ? leaf_buf.pop(): Tensor<FloatType,3>(in_sz,0.);
    assert(in.size(0) == channels);
    assert(in.size(1) == padded_data_len);
    assert(in.size(2) == batch_size);

    int out_data_len = (padded_data_len - kernel_size + stride)/stride;
    int act_der_sz[3] = {depth, out_data_len, batch_size};
    
    Tensor<FloatType,3> activation_deriv = activation_deriv_buf.isFilled() ? activation_deriv_buf.pop() : Tensor<FloatType,3>(act_der_sz,0.);
    assert(activation_deriv.size(0) == depth);
    assert(activation_deriv.size(1) == out_data_len);
    assert(activation_deriv.size(2) == batch_size);      

    //precompute the "activated derivatives" 
    Tensor<FloatType,3> activated_deriv(act_der_sz);
    labelRegionBegin("precompute_activated_derivs");
    {
      autoView(activated_deriv_v,activated_deriv,DeviceWrite);
      autoView(above_deriv_v,above_deriv,DeviceRead);
      autoView(activation_deriv_v,activation_deriv,DeviceRead);
      accelerator_for3d(b,batch_size, o, out_data_len, d,depth,  1, {
	  activated_deriv_v(d,o,b) = above_deriv_v(d,o,b) * activation_deriv_v(d,o,b);
	});
    }      
    labelRegionEnd();

    int depth_=depth;
    int channels_=channels;
    int kernel_size_ =kernel_size;
    int stride_ = stride;
    
    //Compute layer deriv
    //g_do(x) = \sum_c [ \sum_k filter_{dck} x_{c,s*o+k} ]
    //dcost / dx_ci = \sum_do activated_deriv_do [ \sum_k filter_{dck} delta_{s*o+k,i} ]  = \sum_dk activated_deriv_{d,(i-k)/s} filter_{dck} ]
    labelRegionBegin("compute_layer_deriv");
    {
      autoView(layer_deriv_v, layer_deriv, DeviceWrite);
      autoView(activated_deriv_v, activated_deriv, DeviceRead);
      autoView(filter_v, filter, DeviceRead);
      accelerator_for3d(b,batch_size,i,padded_data_len,c,channels,  1, {
	  int kmax = i < kernel_size_-1 ? i : kernel_size_-1;
	  FloatType v=0.;
	  for(int d=0;d<depth_;d++)
	    for(int k=0;k<=kmax;k++)
	      if( (i-k) % stride_ == 0  && (i-k)/stride_ < out_data_len)
		v += activated_deriv_v(d,(i-k)/stride_,b) * filter_v(d,c,k);
	  layer_deriv_v(c,i,b) = v;
	});
    }
    layer_deriv = padding_func.unpadDeriv(layer_deriv); //pass backwards through the padding function
    
    labelRegionEnd();

    //Now we finish up the derivs wrt our parameters
    //g_do(x) = \sum_c [ \sum_k filter_{dck} x_{c,s*o+k} ]
    //
    //dcost / dfilter_{d'c'k'} = \sum_do (dcost/df_do df_do / dg_do) dg_do/filter_{d'c'k'}
    //                      = \sum_o activated_deriv_d'o x_{c',s*o+k'} 
    {
      labelRegionBegin("cost_deriv");
      autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
      autoView(activated_deriv_v, activated_deriv, DeviceRead);
      autoView(in_v,in,DeviceRead);

      accelerator_for2d(b,batch_size, o,out_data_len,  1, {
	  int pp = p;
	  for(int d=0;d<depth_;d++){
	    FloatType act_der_dob = activated_deriv_v(d,o,b);
	    
	    for(int c=0;c<channels_;c++){
	      for(int k=0;k<kernel_size_;k++){
		FloatType v = act_der_dob * in_v(c,stride_*o+k,b);		
		atomicAdd(&cost_deriv_v(pp), v);
		++pp;
	      }
	    }
	  }
	});
      labelRegionEnd();
    }
    p += depth*channels*kernel_size;
    
  }//close views and free temporaries before calling layer below
  profileStop();
  
  return leaf.v.deriv(cost_deriv, p, std::move(layer_deriv), input_above_deriv_return);
}

template<typename FloatType, typename InputType, typename Store, typename ActivationFunc, typename PaddingFunc>
int ConvolutionLayer1D<FloatType,InputType,Store,ActivationFunc,PaddingFunc>::update(int off, const Vector<FloatType> &new_params){
  int p=off;
  {
    autoView(new_params_v,new_params,DeviceRead);
    autoView(filter_v,filter,DeviceWrite);
    int channels_ = channels;
    int kernel_size_ = kernel_size;

    accelerator_for3d(k,kernel_size,c,channels,d,depth,  1, {
	int pp = p + k + kernel_size_*(c + channels_*d);
	filter_v(d,c,k) = new_params_v(pp);
      });

    p+= depth*channels*kernel_size;
  }
  return leaf.v.update(p, new_params);
}

template<typename FloatType, typename InputType, typename Store, typename ActivationFunc, typename PaddingFunc>
int ConvolutionLayer1D<FloatType,InputType,Store,ActivationFunc,PaddingFunc>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  int p=off;
  {
    autoView(derivs_v,derivs,DeviceRead);
    autoView(filter_v,filter,DeviceReadWrite);
    int channels_ = channels;
    int kernel_size_ = kernel_size;

    accelerator_for3d(k,kernel_size,c,channels,d,depth,  1, {
	int pp = p + k + kernel_size_*(c + channels_*d);
	filter_v(d,c,k) -= derivs_v(pp) * eps;
      });

    p+= depth*channels*kernel_size;
  }
  return leaf.v.step(p, derivs, eps);
}



template<typename FloatType, typename InputType, typename Store, typename ActivationFunc, typename PaddingFunc>
int ConvolutionLayer1D<FloatType,InputType,Store,ActivationFunc,PaddingFunc>::getParams(Vector<FloatType> &into, int off){
  int p = off;
  {
    autoView(into_v, into, DeviceReadWrite);
    autoView(filter_v,filter,DeviceRead);
    int channels_ = channels;
    int kernel_size_ = kernel_size;

    accelerator_for3d(k,kernel_size,c,channels,d,depth,  1, {
	int pp = p + k + kernel_size_*(c + channels_*d);
	into_v(pp) = filter_v(d,c,k);
      });

    p+= depth*channels*kernel_size;
  }
  return leaf.v.getParams(into, p);
}
