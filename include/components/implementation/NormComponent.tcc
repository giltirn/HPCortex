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
    autoView(std_store_v,std_store,DeviceWrite);
    
    accelerator_for2d(b,batch_size, o, other_dim_vol, 1, {
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
	  *out_p = ( (*in_p) - mean ) / std;
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
void NormComponent<FloatType,TensDim>::deriv(Tensor<FloatType,TensDim> &&_dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const{
  Tensor<FloatType,TensDim> dcost_by_dOut(std::move(_dcost_by_dOut));
  dcost_by_dIn = Tensor<FloatType,TensDim>(in_size);
  int batch_size = in_size[TensDim-1];
  int norm_dim_size = in_size[norm_dim];
  size_t _stride = stride;
  
  int _norm_dim = norm_dim;
  
  Tensor<FloatType,TensDim> out = out_buf.isFilled() ? out_buf.pop() : Tensor<FloatType,TensDim>(out_buf.latest());
  Matrix<FloatType> stds = std_buf.isFilled() ? std_buf.pop() : Matrix<FloatType>(std_buf.latest());  
  {
    //dCost/dIn_oj = \sum_oj' dCost/dOut_oj' dOut_oj'/dIn_oj
    //Out_oj' = ( In_oj' - mu_o )/sqrt(  var_o + eps )

    //dmu_o/dIn_oj = 1/N
    //dvar_o/dIn_oj =  ds^2_o/dIn_oj - 2*mu_o dmu_oj/dIn_oj
    //s^2_o = \sum_j In_oj^2 / N
    //ds^2_o/dIn_oj = 2 In_oj / N
    
    //dOut_oj' / dIn_oj = 1 /sqrt(  var_o + eps )delta_{jj'}     - dmu_o/dIn_oj /sqrt(  var_o + eps )   - 1/2 * ( In_oj' - mu_o ) / (  var_o + eps )^3/2  dvar_o / dIn_oj
    //                  = 1 /sqrt(  var_o + eps )delta_{jj'}      - 1 / N sqrt(  var_o + eps )     - 1/2 nrm_oj' ( 2 In_oj / N - 2*mu_o / N ) / (  var_o + eps )
    //                  = 1 /sqrt(  var_o + eps )delta_{jj'}      - 1 / N sqrt(  var_o + eps )     - nrm_oj' ( In_oj - mu_o ) / N ( var_o + eps )
    //                  = 1 /sqrt(  var_o + eps )delta_{jj'}      - 1 / N sqrt(  var_o + eps )     - nrm_oj' nrm_oj / N  sqrt( var_o + eps ) 

    //dCost/dIn_oj = \sum_o [ dCost/dOut_oj /sqrt(  var_o + eps )  -\sum_j' dCost/dOut_oj' / N sqrt(  var_o + eps )         - \sum_j' dCost/dOut_oj' nrm_oj' ( In_oj - mu_o ) / N(  var_o + eps )
    
    autoView(out_v,out,DeviceRead);
    autoView(stds_v,stds,DeviceRead);
    autoView(dcost_by_dIn_v, dcost_by_dIn, DeviceWrite);
    autoView(dcost_by_dOut_v, dcost_by_dOut, DeviceRead);
    
    accelerator_for3d(b,batch_size, j,norm_dim_size, o, other_dim_vol, 1, {
	FloatType std = stds_v(o,b);
	size_t off_bo = batchTensorDimensionBaseLin<TensDim>(_norm_dim, b,o, out_v.sizeArray());
	
	FloatType* dcost_by_dOut_p = dcost_by_dOut_v.data() + off_bo;
	FloatType* out_p = out_v.data() + off_bo;
	
	FloatType nrm_oj = out_p[j*_stride];

	FloatType der = dcost_by_dOut_p[j*_stride];
	
	for(int jp=0;jp<norm_dim_size;jp++){
	  FloatType nrm_ojp = *out_p;

	  der -= (*dcost_by_dOut_p) / norm_dim_size;
	  der -= (*dcost_by_dOut_p) * nrm_ojp * nrm_oj / norm_dim_size;
	  
	  dcost_by_dOut_p += _stride;
	  out_p += _stride;
	}
	der = der / std;
	dcost_by_dIn_v.data()[off_bo + _stride * j] = der;	
      });
  }

}
