template<typename Config, int TensDim, typename ActivationFunc>
template<typename InTensorType, enable_if_fwd_ref<InTensorType, Tensor<typename Config::FloatType,TensDim> > >
Tensor<typename Config::FloatType,TensDim> BatchTensorDNNcomponent<Config,TensDim,ActivationFunc>::value(InTensorType &&in, EnableDeriv enable_deriv){
  //INPUT_CON(in,InTensorType);
  
  //DDST(in_ref) in_con(std::forward<InTensorType>(in_ref));
  //auto const &in = in_con.v;
  
  if(!setup){
    batch_size = in.size(TensDim-1);  
    memcpy(in_dims,in.sizeArray(),TensDim*sizeof(int));

    if(in.size(contract_dim) != weights.size(1)){
      std::stringstream ss; ss << "Expected input features " << in.size(contract_dim) << " to match number of columns of weight matrix " << weights.size(1);
      throw std::runtime_error(ss.str());
    }
    
    memcpy(out_dims,in.sizeArray(),TensDim*sizeof(int));
    out_dims[contract_dim] = weights.size(0);

    other_size = 1;
    for(int d=0;d<TensDim-1;d++)
      if(d!= contract_dim)
	other_size *= in_dims[d];

    stride = tensorDimensionStride<TensDim>(contract_dim, in_dims);
    
    setup = true;
  }
  for(int d=0;d<TensDim;d++) assert(in.size(d) == in_dims[d]);

  Tensor<FloatType,TensDim> out = matrixBatchTensorAxpy(weights, in, bias, contract_dim, &value_FLOPS);
  value_FLOPS.lock();
   
  Tensor<FloatType,TensDim> activation_deriv;
  activation_func(out, &activation_deriv);
  for(int i=0;i<TensDim;i++) assert(activation_deriv.size(i) == out.size(i));

  if(enable_deriv){
    in_buf.push(std::forward<InTensorType>(in));
    activation_deriv_buf.push(std::move(activation_deriv));
  }
  return out;    
}

template<typename Config, int TensDim, typename ActivationFunc>
void BatchTensorDNNcomponent<Config,TensDim,ActivationFunc>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, Tensor<FloatType,TensDim> &layer_deriv) const{
  profileStart();
  Tensor<FloatType,TensDim> above_deriv(std::move(_above_deriv));
  for(int i=0;i<TensDim;i++) assert(above_deriv.size(i) == out_dims[i]);
    
  Tensor<FloatType,TensDim> in = in_buf.isFilled() ? in_buf.pop(): Tensor<FloatType,TensDim>(in_dims,0.);
  for(int i=0;i<TensDim;i++) assert(in.size(i) == in_dims[i]);
      
  Tensor<FloatType,TensDim> activation_deriv = activation_deriv_buf.isFilled() ? activation_deriv_buf.pop() : Tensor<FloatType,TensDim>(out_dims,0.);

  //Write output  f_oi(x) = A( g_oi(x) )   where A is the activation function
  //where g_oi(x) = b_i + \sum_j w_ij x_oj
  //
  //dcost / dx_oj = \sum_i dcost/df_oi df_oi/dx_oj
  //df_oi/dx_oj = df_oi / dg_oi dg_oi / dx_oj
  //dg_oi/dx_oj = w_ij
      
  //dcost / dx_oj = \sum_i (dcost/df_oi df_oi/dg_oi) w_ij     :  "layer deriv"
      
  //precompute the "activated derivatives"  (dcost/df_oi df_oi/dg_oi) as they are reused below
  size_t _stride = stride;
  int _contract_dim = contract_dim;
  
  labelRegionBegin("precompute_activated_derivs");
  Tensor<FloatType,TensDim> activated_above_deriv(out_dims);
  {
    if(!deriv_FLOPS.locked()) deriv_FLOPS.add(other_size * out_dims[contract_dim]*batch_size); 
    
    autoView(activated_above_deriv_v, activated_above_deriv, DeviceWrite);
    autoView(above_deriv_v, above_deriv, DeviceRead);
    autoView(activation_deriv_v, activation_deriv, DeviceRead);    
    accelerator_for_3d_gen(1,2,normal(),  b, batch_size, i, out_dims[contract_dim], o, other_size, {
	size_t poff = batchTensorDimensionBaseLin<TensDim>(_contract_dim, b, o, above_deriv_v.sizeArray())   + i*_stride;
	activated_above_deriv_v.data()[poff] = above_deriv_v.data()[poff] * activation_deriv_v.data()[poff];
      });
  }
  labelRegionEnd();
      
  //Compute layer deriv
  //dcost / dx_oj = \sum_i (dcost/df_oi df_oi/dg_oi) w_ij =\sum_i activated_deriv_oi w_ij 
  labelRegionBegin("compute_layer_deriv");
  layer_deriv = matrixBatchTensorContractRight(activated_above_deriv, weights, contract_dim, &deriv_FLOPS);
  labelRegionEnd();
  
  //Now we finish up the derivs wrt our parameters      
  //dcost / dw_jk = \sum_oi (dcost/df_oi df_oi/dg_oi) dg_oi/dw_jk
  //dcost / db_j = \sum_oi (dcost/df_oi df_oi/dg_oi) dg_oi/db_j
  
  //dg_oi / d w_jk = delta_ij x_ok
  //dg_oi / d b_j = delta_ij
  
  //dcost / dw_jk = \sum_o (dcost/df_oj df_oj/dg_oj) x_ok  =  \sum_o activated_deriv_oj x_ok
  //dcost / d b_j = \sum_o (dcost/df_oj df_oj/dg_oj) = \sum_o activated_deriv_oj
  {
    int p=off;
    
    labelRegionBegin("cost_deriv_weights");
    {
      autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
      batchTensorContractToMatrix_p(cost_deriv_v.data() + p, activated_above_deriv, in, contract_dim, &deriv_FLOPS);
    }
  
    p += weights.size(0)*weights.size(1);
    labelRegionEnd();

    if(use_bias){
      if(!deriv_FLOPS.locked()) deriv_FLOPS.add(weights.size(0) * other_size * batch_size);
      
      size_t bs = batch_size;
      autoView(activated_above_deriv_v, activated_above_deriv, DeviceRead);
      autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
      labelRegionBegin("cost_deriv_bias");

      accelerator_for_2d_gen(1,1,splitBlock<64>(), j, weights.size(0), o, other_size, {
	  FloatType* activated_above_deriv_p = activated_above_deriv_v.data() + batchTensorDimensionBaseLin<TensDim>(_contract_dim, 0, o, activated_above_deriv_v.sizeArray()) + _stride*j;

	  FloatType v = *activated_above_deriv_p++;
	  for(int b=1;b<bs;b++)
	    v += *activated_above_deriv_p++;
	  atomicAdd( cost_deriv_v.data() + p + j , v );
	});

      p += weights.size(0);    
      labelRegionEnd();
    }
    
  }
  deriv_FLOPS.lock();
  profileStop();
}

template<typename Config, int TensDim, typename ActivationFunc>
void BatchTensorDNNcomponent<Config,TensDim,ActivationFunc>::update(int off, const Vector<FloatType> &new_params){
  autoView(new_params_v, new_params,DeviceRead);
  FloatType const* dp = new_params_v.data() + off;
  {
    autoView(weights_v,weights,DeviceWrite);    
    accelerator_for_gen(1,0,splitBlock<32>(),o,weights_v.data_len(), {
	weights_v.data()[o] = dp[o];
      });
  }

  if(use_bias){
    dp += weights.data_len();
    
    autoView(bias_v,bias,DeviceWrite);
    accelerator_for_gen(1,0,splitBlock<32>(),o,bias.size(0), {
	bias_v.data()[o] = dp[o];
      });	
  }
}

template<typename Config, int TensDim, typename ActivationFunc>
void BatchTensorDNNcomponent<Config,TensDim,ActivationFunc>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  autoView(derivs_v,derivs,DeviceRead);
  FloatType const* dp = derivs_v.data() + off;
  {
    autoView(weights_v,weights,DeviceReadWrite);    
    accelerator_for_gen(1,0,splitBlock<32>(),o,weights_v.data_len(), {
	weights_v.data()[o] -= dp[o]*eps;
      });
  }

  if(use_bias){
    dp += weights.data_len();
    
    autoView(bias_v,bias,DeviceReadWrite);
    accelerator_for_gen(1,0,splitBlock<32>(),o,bias.size(0), {
	bias_v.data()[o] -= dp[o]*eps;
      });	
  }
}

template<typename Config, int TensDim, typename ActivationFunc>
void BatchTensorDNNcomponent<Config,TensDim,ActivationFunc>::getParams(Vector<FloatType> &into, int off) const{
  autoView(into_v,into,DeviceReadWrite);
  FloatType * dp = into_v.data() + off;
  
  {
    autoView(weights_v,weights,DeviceRead);    
    accelerator_for_gen(1,0,splitBlock<32>(),o,weights_v.data_len(), {
	dp[o] = weights_v.data()[o];
      });
  }

  if(use_bias){
    dp += weights.data_len();
    
    autoView(bias_v,bias,DeviceRead);
    accelerator_for_gen(1,0,splitBlock<32>(),o,bias.size(0), {
	dp[o] = bias_v.data()[o];
      });	
  }
}

