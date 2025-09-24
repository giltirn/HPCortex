template<typename Config, int TensDim>
template<typename InTensorType, enable_if_fwd_ref<InTensorType,Tensor<typename Config::FloatType,TensDim> > >
Tensor<typename Config::FloatType,TensDim> MatrixTensorContractComponent<Config,TensDim>::value(InTensorType &&in_ref, EnableDeriv enable_deriv){
  INPUT_CON(in, InTensorType);
  
  if(!setup || in.size(TensDim-1) != batch_size ){
    batch_size = in.size(TensDim-1);  
    memcpy(in_dims,in.sizeArray(),TensDim*sizeof(int));
  
    memcpy(out_dims,in.sizeArray(),TensDim*sizeof(int));
    out_dims[TensDim-2] = size0;
    
    setup = true;
  } 

  if(in.size(TensDim-2) != size1){
    std::stringstream ss; ss << "Expected input features " << in.size(TensDim-2) << " to match number of columns of weight matrix " << size1;
    throw std::runtime_error(ss.str());
  }

  Tensor<FloatType,TensDim> out = matrixBatchTensorContractLeft(weights, in, TensDim-2, &value_FLOPS);
  value_FLOPS.lock();
  
  if(enable_deriv) in_buf.push(in_con.release());
  return out;    
}

template<typename Config, int TensDim>
void MatrixTensorContractComponent<Config,TensDim>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, Tensor<FloatType,TensDim> &layer_deriv) const{
  assert(_above_deriv.size(TensDim-2) == size0);
  assert(_above_deriv.size(TensDim-1) == batch_size);

  //f_oi = \sum_j w_ij x_oj      for compound index o
  //dcost / dx_oj = \sum_i dcost/df_oi *  df_oi / dx_oj   : "layer_deriv" 
  //df_oi / dx_oj = w_ij 
  //dcost / dx_oj = \sum_i dcost/df_oi *  w_ij

  //dcost / dw_ij = \sum_i'o   dcost/df_oi'  *  df_oi' / dw_ij 
  //df_oi' / dw_ij = x_oj \delta_i'i
  //dcost / dw_ij = \sum_o dcost/df_oi  x_oj

  Tensor<FloatType,TensDim> above_deriv(std::move(_above_deriv)); //inside the braces above ensures this object is freed before the next layer is called

  Tensor<FloatType,TensDim> in = in_buf.isFilled() ? in_buf.pop(): Tensor<FloatType,TensDim>(in_dims,0.);
  assert(in.size(TensDim-2) == size1);
  assert(in.size(TensDim-1) == batch_size);

  //compute layer deriv
  //dcost / dx_oj = \sum_i dcost/df_oi *  w_ij
  layer_deriv = matrixBatchTensorContractRight(above_deriv, weights, TensDim-2, &deriv_FLOPS);

  //compute cost_deriv
  //dcost / dw_ij = \sum_o dcost/df_oi  x_oj
  {
    autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
    batchTensorContractToMatrix_p(cost_deriv_v.data() + off, above_deriv, in, TensDim-2, &deriv_FLOPS);
  }
  deriv_FLOPS.lock();
}

template<typename Config, int TensDim>
void MatrixTensorContractComponent<Config,TensDim>::update(int off, const Vector<FloatType> &new_params){
  autoView(new_params_v,new_params,DeviceRead);
  autoView(weights_v,weights,DeviceWrite);
  size_t sz1=size1;
  accelerator_for2d(j,size1,i,size0,1,{
      int pp = off + j + sz1*i;
      weights_v(i,j) = new_params_v(pp);
    });
}

template<typename Config, int TensDim>
void MatrixTensorContractComponent<Config,TensDim>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  autoView(derivs_v,derivs,DeviceRead);
  autoView(weights_v,weights,DeviceReadWrite);
  size_t sz1 = size1;
  accelerator_for2d(j,size1,i,size0,1,{
      int pp = off + j + sz1*i;
      weights_v(i,j) -= derivs_v(pp) * eps;
    });
	  
}


template<typename Config, int TensDim>
void MatrixTensorContractComponent<Config,TensDim>::getParams(Vector<FloatType> &into, int off) const{
  autoView(into_v,into,DeviceReadWrite);
  autoView(weights_v,weights,DeviceRead);
  size_t sz1 = size1;
  accelerator_for2d(j,size1,i,size0,1,{
      int pp = off + j + sz1*i;
      into_v(pp) = weights_v(i,j);
    });
}
