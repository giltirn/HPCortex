template<typename Config, int TensDim>
Tensor<typename Config::FloatType,TensDim> ScaleComponent<Config,TensDim>::value(const Tensor<FloatType,TensDim> &in, EnableDeriv enable_deriv){
  if(!setup){
    assert(in.size(scale_dim) == beta.size(0));
    
    memcpy(in_size, in.sizeArray(), TensDim*sizeof(int));
    other_dim_vol = 1;
    for(int d=0;d<TensDim-1;d++)
      if(d!=scale_dim)
	other_dim_vol *= in.size(d);

    stride = tensorDimensionStride<TensDim>(scale_dim,in.sizeArray());
    setup = true;
  }
  
  int batch_size = in.size(TensDim-1);
  int scale_dim_size = in.size(scale_dim);
  size_t _stride = stride;
  
  Tensor<FloatType,TensDim> out(in.sizeArray());
  int _scale_dim = scale_dim;

  if(!value_FLOPS.locked()){
    value_FLOPS.add(other_dim_vol*scale_dim_size*batch_size*2);
    value_FLOPS.lock();
  }
  
  {
    autoView(in_v,in,DeviceRead);
    autoView(out_v,out,DeviceWrite);
    autoView(gamma_v,gamma,DeviceRead);
    autoView(beta_v,beta,DeviceRead);
    
    accelerator_for3d(b,batch_size, i, scale_dim_size, o, other_dim_vol, 1, {
	size_t off = batchTensorDimensionBaseLin<TensDim>(_scale_dim, b,o, in_v.sizeArray()) + i*_stride;      
	FloatType* in_p = in_v.data() + off;
	FloatType* out_p = out_v.data() + off;
      	
	(*out_p) = gamma_v(i)*(*in_p) + beta_v(i);	  
      });
  }
  
  if(enable_deriv) in_buf.push(Tensor<FloatType,TensDim>(in)); //TODO: allow pass by r-value and move here
  
  return out;		  
}

template<typename Config, int TensDim>
void ScaleComponent<Config,TensDim>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const{
  Tensor<FloatType,TensDim> dcost_by_dOut(std::move(_dcost_by_dOut));
  dcost_by_dIn = Tensor<FloatType,TensDim>(in_size);
  int batch_size = in_size[TensDim-1];
  int scale_dim_size = in_size[scale_dim];
  size_t _stride = stride;
  
  bool _use_affine = use_affine;
  bool _use_bias = use_bias;
  int _scale_dim = scale_dim;
 
  Tensor<FloatType,TensDim> in = in_buf.isFilled() ? in_buf.pop() : Tensor<FloatType,TensDim>(in_buf.latest());
  {
    //dCost/dIn_oj = dCost/dOut_oj dOut_oj/dIn_oj
    //Out_oj = gamma_j * In_oj + beta_j
    //dOut_oj/dIn_oj = gamma_j
    //dCost/dIn_oj = dCost/dOut_oj gamma_j
    if(!deriv_FLOPS.locked()) deriv_FLOPS.add(other_dim_vol*scale_dim_size*batch_size);
    
    autoView(dcost_by_dIn_v, dcost_by_dIn, DeviceWrite);
    autoView(dcost_by_dOut_v, dcost_by_dOut, DeviceRead);
    autoView(gamma_v,gamma, DeviceRead);
    
    accelerator_for3d(b,batch_size, j, scale_dim_size, o, other_dim_vol, 1, {
	size_t off_boj = batchTensorDimensionBaseLin<TensDim>(_scale_dim, b,o, dcost_by_dOut_v.sizeArray()) + _stride * j;
	
	FloatType* dcost_by_dOut_p = dcost_by_dOut_v.data() + off_boj;
	FloatType* dcost_by_dIn_p = dcost_by_dIn_v.data() + off_boj;
	(*dcost_by_dIn_p) = (*dcost_by_dOut_p) * gamma_v(j);
      });
  }
  if(use_affine || use_bias){
    //dOut_oj/dgamma_j = In_oj
    //dCost/dgamma_j = \sum_o dCost/dOut_oj In_oj
    
    //dOut_oj/dbeta_j = 1
    //dCost/dbeta_j = \sum_o dCost/dOut_oj

    if(!deriv_FLOPS.locked()) deriv_FLOPS.add(
					      other_dim_vol*scale_dim_size*(
									    (use_affine ? batch_size*2 + 1 : 0)
									    +
									    (use_bias ? batch_size + 1 : 0)
									    )
					      );
    
    autoView(cost_deriv_v, cost_deriv, DeviceReadWrite);
    autoView(dcost_by_dOut_v, dcost_by_dOut, DeviceRead);
    autoView(in_v,in,DeviceRead);
    
    autoView(gamma_v,gamma,DeviceRead);
    autoView(beta_v, beta,DeviceRead);

    int p_gamma = off;
    int p_beta = use_affine ? off + scale_dim_size : off;
   
    accelerator_for_2d_gen(1,1,splitBlock<64>(), j, scale_dim_size, o, other_dim_vol, {
	size_t off_0oj = batchTensorDimensionBaseLin<TensDim>(_scale_dim, 0,o, dcost_by_dOut_v.sizeArray()) + j*_stride;
	FloatType* dcost_by_dOut_p = dcost_by_dOut_v.data() + off_0oj;
	FloatType* in_p = in_v.data() + off_0oj;

	FloatType der_gamma = 0., der_beta=0.;
	for(int b=0;b<batch_size;b++){
	  der_gamma += (*dcost_by_dOut_p) * (*in_p++);
	  der_beta += (*dcost_by_dOut_p++);
	}
	if(_use_affine) atomicAdd(&cost_deriv_v(p_gamma + j), der_gamma); //sum_o
	if(_use_bias) atomicAdd(&cost_deriv_v(p_beta + j), der_beta);
      });
  }
  deriv_FLOPS.lock();
}

template<typename Config, int TensDim>
void ScaleComponent<Config,TensDim>::update(int off, const Vector<FloatType> &new_params){
  if(!use_affine && !use_bias) return;
  bool _use_affine = use_affine;
  bool _use_bias = use_bias;
  int p_gamma = off;
  int p_beta = use_affine ? off + gamma.size(0) : off;
  
  autoView(new_params_v,new_params,DeviceRead);
  autoView(gamma_v,gamma,DeviceWrite);
  autoView(beta_v,beta,DeviceWrite);
  accelerator_for(i,gamma.size(0), {
      if(_use_affine) gamma_v(i) = new_params_v(p_gamma+i);
      if(_use_bias) beta_v(i) = new_params_v(p_beta+i);
    });
}
  
  
template<typename Config, int TensDim>
void ScaleComponent<Config,TensDim>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  if(!use_affine && !use_bias) return;
  bool _use_affine = use_affine;
  bool _use_bias = use_bias;
  int p_gamma = off;
  int p_beta = use_affine ? off + gamma.size(0) : off;
  
  autoView(derivs_v,derivs,DeviceRead);
  autoView(gamma_v,gamma,DeviceReadWrite);
  autoView(beta_v,beta,DeviceReadWrite);
  accelerator_for(i,gamma.size(0), {
      if(_use_affine) gamma_v(i) -= derivs_v(p_gamma+i) * eps;
      if(_use_bias) beta_v(i) -= derivs_v(p_beta+i) * eps;
    });

}
  
template<typename Config, int TensDim>
void ScaleComponent<Config,TensDim>::getParams(Vector<FloatType> &into, int off) const{
  if(!use_affine && !use_bias) return;
  bool _use_affine = use_affine;
  bool _use_bias = use_bias;
  int p_gamma = off;
  int p_beta = use_affine ? off + gamma.size(0) : off;
  
  autoView(into_v,into,DeviceReadWrite);
  autoView(gamma_v,gamma,DeviceReadWrite);
  autoView(beta_v,beta,DeviceReadWrite);
  accelerator_for(i,gamma.size(0), {
      if(_use_affine) into_v(p_gamma+i) = gamma_v(i);
      if(_use_bias) into_v(p_beta+i) = beta_v(i);
    });
}
