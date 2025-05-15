template<typename FloatType, int TensDim, typename ActivationFunc>
Tensor<FloatType,TensDim> BatchTensorDNNcomponent<FloatType,TensDim,ActivationFunc>::value(const Tensor<FloatType,TensDim> &in){
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
   
  Tensor<FloatType,TensDim> out(out_dims); 
  {
    autoView(in_v, in, DeviceRead);
    autoView(weights_v, weights, DeviceRead);
    autoView(bias_v, bias, DeviceRead);
    autoView(out_v, out, DeviceWrite);
    
    int _sizej = weights.size(1);
    int _sizei = weights.size(0);
    int _contract_dim = contract_dim;
    size_t _stride = stride;

    //W_{ij} X_{..., j, ..., b}  
    accelerator_for3d(b, batch_size, i, _sizei, o, other_size,    1, {
	size_t off_in = batchTensorDimensionBaseLin<TensDim>(_contract_dim, b, o, in_v.sizeArray());
	size_t off_out = batchTensorDimensionBaseLin<TensDim>(_contract_dim, b, o, out_v.sizeArray());
	FloatType *in_p = in_v.data() + off_in;
	
	FloatType out_oib = weights_v(i,0) * (*in_p);
	in_p += _stride;
	
	for(int j=1;j<_sizej;j++){
	  out_oib += weights_v(i,j) * (*in_p);
	  in_p += _stride;
	}	  
	
	out_v.data()[off_out + _stride*i] = out_oib  + bias_v(i);
      });
  }
   
  in_buf.push(Tensor<FloatType,TensDim>(in)); //TODO: Can avoid this copy in some case by allowing r-value references for inputs. Perhaps have 2 versions of "value", taking l-value and r-value refs, respectively?
  
  Tensor<FloatType,TensDim> activation_deriv;
  activation_func(out, &activation_deriv);
  for(int i=0;i<TensDim;i++) assert(activation_deriv.size(i) == out.size(i));
  
  activation_deriv_buf.push(std::move(activation_deriv));
  
  return out;    
}

template<typename FloatType, int TensDim, typename ActivationFunc>
void BatchTensorDNNcomponent<FloatType,TensDim,ActivationFunc>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, Tensor<FloatType,TensDim> &layer_deriv) const{
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
    autoView(activated_above_deriv_v, activated_above_deriv, DeviceWrite);
    autoView(above_deriv_v, above_deriv, DeviceRead);
    autoView(activation_deriv_v, activation_deriv, DeviceRead);    
    accelerator_for3d(b, batch_size, i, out_dims[contract_dim], o, other_size, 1, {
	size_t poff = batchTensorDimensionBaseLin<TensDim>(_contract_dim, b, o, above_deriv_v.sizeArray())   + i*_stride;
	activated_above_deriv_v.data()[poff] = above_deriv_v.data()[poff] * activation_deriv_v.data()[poff];
      });
  }
  labelRegionEnd();
      
  //Compute layer deriv
  //dcost / dx_oj = \sum_i (dcost/df_oi df_oi/dg_oi) w_ij =\sum_i activated_deriv_oi w_ij 
  labelRegionBegin("compute_layer_deriv");
  {
    layer_deriv = Tensor<FloatType,TensDim>(in_dims);
    autoView(layer_deriv_v, layer_deriv, DeviceWrite);
    autoView(weights_v, weights, DeviceRead);
    autoView(activated_above_deriv_v, activated_above_deriv, DeviceRead);

    int sizei = weights.size(0);
    accelerator_for3d(b, batch_size, j, weights.size(1), o, other_size, 1, {
	size_t out_poff = batchTensorDimensionBaseLin<TensDim>(_contract_dim, b, o, layer_deriv_v.sizeArray());
	size_t in_poff =  batchTensorDimensionBaseLin<TensDim>(_contract_dim, b, o, activated_above_deriv_v.sizeArray());

	FloatType* activated_above_deriv_p =  activated_above_deriv_v.data() + in_poff; //activated derivatives_{o,0}
	FloatType v = (*activated_above_deriv_p)*weights_v(0,j);
	activated_above_deriv_p += _stride;
	
	for(int i=1;i<sizei;i++){
	  v +=(*activated_above_deriv_p)*weights_v(i,j);
	  activated_above_deriv_p += _stride;
	}
	layer_deriv_v.data()[out_poff + j*_stride] = v;
      });
  }
  labelRegionEnd();
  
  //Now we finish up the derivs wrt our parameters      
  //dcost / dw_jk = \sum_oi (dcost/df_oi df_oi/dg_oi) dg_oi/dw_jk
  //dcost / db_j = \sum_oi (dcost/df_oi df_oi/dg_oi) dg_oi/db_j
  
  //dg_oi / d w_jk = delta_ij x_ok
  //dg_oi / d b_j = delta_ij
  
  //dcost / dw_jk = \sum_o (dcost/df_oj df_oj/dg_oj) x_ok  =  \sum_o activated_deriv_oj x_ok 
  {
    int p=off;
    
    labelRegionBegin("cost_deriv_weights");
    size_t bs = batch_size;
    autoView(activated_above_deriv_v, activated_above_deriv, DeviceRead);
    autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
    autoView(in_v,in,DeviceRead);

    int sizek = weights.size(1);
    accelerator_for3d(dummy,1, jk, weights.size(0)*weights.size(1), o, other_size, 64,{ 
	int k = jk % sizek;
	int j = jk / sizek;  //jk = k+sizek*j
	//Sum over batch index, neighboring in memory
	FloatType* activated_above_deriv_p = activated_above_deriv_v.data() + batchTensorDimensionBaseLin<TensDim>(_contract_dim, 0, o, activated_above_deriv_v.sizeArray()) + _stride*j;
	FloatType* in_p = in_v.data() + batchTensorDimensionBaseLin<TensDim>(_contract_dim, 0, o, in_v.sizeArray()) + _stride*k;

	FloatType v = (*activated_above_deriv_p++) * (*in_p++);
	for(int b=1;b<bs;b++)
	  v += (*activated_above_deriv_p++) * (*in_p++);
	atomicAdd(cost_deriv_v.data() + p + jk,  v); //sum over o
      });	         
	
    p += weights.size(0)*weights.size(1);
    labelRegionEnd();

    if(use_bias){
      labelRegionBegin("cost_deriv_bias");

      accelerator_for3d(dummy1,1, j, weights.size(0), o, other_size, 64,{
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
  profileStop();
}

template<typename FloatType, int TensDim, typename ActivationFunc>
void BatchTensorDNNcomponent<FloatType,TensDim,ActivationFunc>::update(int off, const Vector<FloatType> &new_params){
  autoView(new_params_v,new_params,DeviceRead);
  int p = off;
  {
    autoView(weights_v,weights,DeviceWrite);
    accelerator_for3d(dummy1,1,j,weights.size(1),i,weights.size(0),64,{
	int pp = p + j + weights_v.size(1)*i;
	weights_v(i,j) = new_params_v(pp);
      });
    p += weights.size(0)*weights.size(1);
  }
  if(use_bias){
    autoView(bias_v,bias,DeviceWrite);
    accelerator_for2d(dummy1,1,i,weights.size(0),64,{
	int pp = p + i;
	bias_v(i) = new_params_v(pp);
      });
  }
}

template<typename FloatType, int TensDim, typename ActivationFunc>
void BatchTensorDNNcomponent<FloatType,TensDim,ActivationFunc>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  autoView(derivs_v,derivs,DeviceRead);
  int p = off;
  {
    autoView(weights_v,weights,DeviceReadWrite);
    accelerator_for3d(dummy1,1,j,weights.size(1),i,weights.size(0),64,{
	int pp = p + j + weights_v.size(1)*i;
	weights_v(i,j) -= derivs_v(pp)*eps;
    });
    p += weights.size(0)*weights.size(1);
  }
  if(use_bias){
    autoView(bias_v,bias,DeviceReadWrite);
    accelerator_for2d(dummy1,1,i,weights.size(0),64,{
      int pp = p + i;
      bias_v(i) -= derivs_v(pp)*eps;
    });
  }
}



template<typename FloatType, int TensDim, typename ActivationFunc>
void BatchTensorDNNcomponent<FloatType,TensDim,ActivationFunc>::getParams(Vector<FloatType> &into, int off){
  autoView(into_v,into,DeviceReadWrite);
  int p = off;
  {
    autoView(weights_v,weights,DeviceRead);
    accelerator_for3d(dummy1,1,j,weights.size(1),i,weights.size(0),64,{
	int pp = p + j + weights_v.size(1)*i;
	into_v(pp) = weights_v(i,j);
    });
    p += weights.size(0)*weights.size(1);
  }
  if(use_bias){
    autoView(bias_v,bias,DeviceReadWrite);
    accelerator_for2d(dummy1,1,i,weights.size(0),64,{
      int pp = p + i;
      into_v(pp) = bias_v(i);
    });
  }
}

