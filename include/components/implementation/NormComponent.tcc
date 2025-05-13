template<typename FloatType, int TensDim>
Tensor<FloatType,TensDim> NormComponent<FloatType,TensDim>::value(const Tensor<FloatType,TensDim> &in){
  if(!setup){
    memcpy(in_size, in.sizeArray(), TensDim*sizeof(int));
    other_dim_vol = 1;
    for(int d=0;d<TensDim-1;d++)
      if(d!=norm_dim)
	other_dim_vol *= in.size(d);

    stride = tensorDimensionStride<TensDim>(norm_dim,in.sizeArray());
    setup = true;
  }
  
  int batch_size = in.size(TensDim-1);
  int norm_dim_size = in.size(norm_dim);
  size_t _stride = stride;
  
  FloatType _epsilon = epsilon;
  
  Tensor<FloatType,TensDim> out(in.sizeArray());
  Matrix<FloatType> std_store(other_dim_vol, batch_size);
  
  int _norm_dim = norm_dim;
  
  {
    autoView(in_v,in,DeviceRead);
    autoView(out_v,out,DeviceWrite);
    autoView(gamma_beta_v,gamma_beta,DeviceRead);
    autoView(std_store_v,std_store,DeviceWrite);
    
    accelerator_for2d(b,batch_size, o, other_dim_vol, 1, {
	FloatType _affine_gamma = gamma_beta_v(0);
	FloatType _bias_beta = gamma_beta_v(1);

	size_t off = batchTensorDimensionBaseLin<TensDim>(_norm_dim, b,o, in_v.sizeArray());      
	FloatType* in_p = in_v.data() + off;
      
	FloatType mean = 0.;
	FloatType var = 0.;
	
	for(int i=0;i<norm_dim_size;i++){
	  FloatType ii = *in_p; in_p += _stride;
	  mean += ii;
	  var += ii*ii;
	}
	mean = mean / norm_dim_size;
	var = var / norm_dim_size - mean*mean;

	FloatType std = sqrt(var + _epsilon);
	
	in_p = in_v.data() + off;
	FloatType* out_p = out_v.data() + off;
      
	for(int i=0;i<norm_dim_size;i++){
	  *out_p = ( (*in_p) - mean ) / std * _affine_gamma + _bias_beta;
	  out_p += _stride;
	  in_p += _stride;
	}

	std_store_v(o,b) = std;
      });
  }
  
  out_buf.push(Tensor<FloatType,TensDim>(out));
  std_buf.push(std::move(std_store));
  
  return out;		  
}

template<typename FloatType, int TensDim>
void NormComponent<FloatType,TensDim>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const{
  Tensor<FloatType,TensDim> dcost_by_dOut(std::move(_dcost_by_dOut));
  dcost_by_dIn = Tensor<FloatType,TensDim>(in_size);
  int batch_size = in_size[TensDim-1];
  int norm_dim_size = in_size[norm_dim];
  size_t _stride = stride;
  
  bool _use_affine = use_affine;
  bool _use_bias = use_bias;
  int _norm_dim = norm_dim;

  int p_gamma = off;
  int p_beta = use_affine ? off + 1 : off;
  
  Tensor<FloatType,TensDim> out = out_buf.isFilled() ? out_buf.pop() : Tensor<FloatType,TensDim>(out_buf.latest());
  Matrix<FloatType> stds = std_buf.isFilled() ? std_buf.pop() : Matrix<FloatType>(std_buf.latest());  
  {
    //dCost/dIn_oj = \sum_oj' dCost/dOut_oj' dOut_oj'/dIn_oj
    //Out_oj' = gamma * nrm_oj' + beta
    //nrm_oj' = ( In_oj' - mu_o )/sqrt(  var_o + eps )

    //dmu_o/dIn_oj = 1/N
    //dvar_o/dIn_oj =  ds^2_o/dIn_oj - 2*mu_o dmu_oj/dIn_oj
    //s^2_o = \sum_j In_oj^2 / N
    //ds^2_o/dIn_oj = 2 In_oj / N
    
    //dOut_oj' / dIn_oj = gamma /sqrt(  var_o + eps )delta_{jj'}     - gamma dmu_o/dIn_oj /sqrt(  var_o + eps )   - 1/2 gamma * ( In_oj' - mu_o ) / (  var_o + eps )^3/2  dvar_o / dIn_oj
    //                 = gamma /sqrt(  var_o + eps )delta_{jj'}      - gamma / N sqrt(  var_o + eps )     - 1/2 gamma nrm_oj' ( 2 In_oj / N - 2*mu_o / N ) / (  var_o + eps )
    //                 = gamma /sqrt(  var_o + eps )delta_{jj'}      - gamma / N sqrt(  var_o + eps )     - gamma nrm_oj' ( In_oj - mu_o ) / N ( var_o + eps )
    //                 = gamma /sqrt(  var_o + eps )delta_{jj'}      - gamma / N sqrt(  var_o + eps )     - gamma nrm_oj' nrm_oj / N  sqrt( var_o + eps ) 

    //dCost/dIn_oj = \sum_o [ dCost/dOut_oj gamma/sqrt(  var_o + eps )  -\sum_j' dCost/dOut_oj' gamma / N sqrt(  var_o + eps )         - \sum_j' dCost/dOut_oj' nrm_oj' ( In_oj - mu_o ) / N(  var_o + eps )
    
    autoView(out_v,out,DeviceRead);
    autoView(stds_v,stds,DeviceRead);
    autoView(dcost_by_dIn_v, dcost_by_dIn, DeviceWrite);
    autoView(dcost_by_dOut_v, dcost_by_dOut, DeviceRead);
    autoView(gamma_beta_v,gamma_beta, DeviceRead);
    
    accelerator_for3d(b,batch_size, j,norm_dim_size, o, other_dim_vol, 1, {
	FloatType _affine_gamma = gamma_beta_v(0);
	FloatType _bias_beta = gamma_beta_v(1);
	
	FloatType std = stds_v(o,b);
	size_t off_bo = batchTensorDimensionBaseLin<TensDim>(_norm_dim, b,o, out_v.sizeArray());
	
	FloatType* dcost_by_dOut_p = dcost_by_dOut_v.data() + off_bo;
	FloatType* out_p = out_v.data() + off_bo;
	
	FloatType nrm_oj = ( out_p[j*_stride] - _bias_beta ) / _affine_gamma;

	FloatType der = dcost_by_dOut_p[j*_stride];
	
	for(int jp=0;jp<norm_dim_size;jp++){
	  FloatType nrm_ojp = ( (*out_p) - _bias_beta ) / _affine_gamma;

	  der -= (*dcost_by_dOut_p) / norm_dim_size;
	  der -= (*dcost_by_dOut_p) * nrm_ojp * nrm_oj / norm_dim_size;
	  
	  dcost_by_dOut_p += _stride;
	  out_p += _stride;
	}
	der = der * _affine_gamma / std;
	dcost_by_dIn_v.data()[off_bo + _stride * j] = der;	
      });
  }
  if(use_affine || use_bias){
    //dCost/dgamma = \sum_oj' dCost/dOut_oj' dOut_oj'/dgamma
    //             = \sum_oj' dCost/dOut_oj' nrm_oj'
    //dCost/dbeta = \sum_oj' dCost/dOut_oj' dOut_oj'/dbeta
    //             = \sum_oj' dCost/dOut_oj'
        
    autoView(out_v,out,DeviceRead);
    autoView(cost_deriv_v, cost_deriv, DeviceReadWrite);
    autoView(dcost_by_dOut_v, dcost_by_dOut, DeviceRead);
    autoView(gamma_beta_v,gamma_beta,DeviceRead);
        
    accelerator_for2d_shm(b,batch_size, o, other_dim_vol, 1, (2*norm_dim_size*sizeof(FloatType)), {
	extern __shared__ FloatType shared[];
	FloatType _affine_gamma = gamma_beta_v(0);
	FloatType _bias_beta = gamma_beta_v(1);
	
	size_t off_bo = batchTensorDimensionBaseLin<TensDim>(_norm_dim, b,o, dcost_by_dOut_v.sizeArray());
	FloatType* dcost_by_dOut_p = dcost_by_dOut_v.data() + off_bo;
	FloatType* out_p = out_v.data() + off_bo;

	FloatType der_gamma = 0., der_beta = 0.;
	for(int j=0;j<norm_dim_size;j++){
	  FloatType nrm_oj = ( (*out_p) - _bias_beta ) / _affine_gamma;
	  
      	  der_gamma += (*dcost_by_dOut_p) * nrm_oj;
	  der_beta +=  (*dcost_by_dOut_p);

	  dcost_by_dOut_p += _stride;
	  out_p += _stride;
	}

	shared[2*b] = der_gamma;
	shared[2*b+1] = der_beta;
	acceleratorSynchronizeBlock();

	if(b==0){  
	  FloatType sma = 0, smb = 0;
	  FloatType* sp = shared;
	  for(int b=0;b<batch_size;b++){
	    sma += *(sp++); //sum_b
	    smb += *(sp++);
	  }	  
	  if(_use_affine) atomicAdd(&cost_deriv_v(p_gamma), sma);
	  if(_use_bias) atomicAdd(&cost_deriv_v(p_beta), smb);
	}
      });
  }

}

template<typename FloatType, int TensDim>
void NormComponent<FloatType,TensDim>::update(int off, const Vector<FloatType> &new_params){
  if(!use_affine && !use_bias) return;
  bool _use_affine = use_affine;
  bool _use_bias = use_bias;
  int p_gamma = off;
  int p_beta = use_affine ? off + 1 : off;
  
  autoView(new_params_v,new_params,DeviceRead);
  autoView(gamma_beta_v,gamma_beta,DeviceWrite);
  accelerator_for(i,1, {
      if(_use_affine) gamma_beta_v(0) = new_params_v(p_gamma);
      if(_use_bias) gamma_beta_v(1) = new_params_v(p_beta);
    });
}
  
  
template<typename FloatType, int TensDim>
void NormComponent<FloatType,TensDim>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  if(!use_affine && !use_bias) return;
  bool _use_affine = use_affine;
  bool _use_bias = use_bias;
  int p_gamma = off;
  int p_beta = use_affine ? off + 1 : off;
  
  autoView(derivs_v,derivs,DeviceRead);
  autoView(gamma_beta_v,gamma_beta,DeviceReadWrite);
  accelerator_for(i,1, {
      if(_use_affine) gamma_beta_v(0) -= derivs_v(p_gamma) * eps;
      if(_use_bias) gamma_beta_v(1) -= derivs_v(p_beta) * eps;
    });
}
  
template<typename FloatType, int TensDim>
void NormComponent<FloatType,TensDim>::getParams(Vector<FloatType> &into, int off){
  if(!use_affine && !use_bias) return;
  bool _use_affine = use_affine;
  bool _use_bias = use_bias;
  int p_gamma = off;
  int p_beta = use_affine ? off + 1 : off;
  
  autoView(into_v,into,DeviceReadWrite);
  autoView(gamma_beta_v,gamma_beta,DeviceRead);
  accelerator_for(i,1, {
      if(_use_affine) into_v(p_gamma) = gamma_beta_v(0);
      if(_use_bias) into_v(p_beta) = gamma_beta_v(1);
    });  
}
