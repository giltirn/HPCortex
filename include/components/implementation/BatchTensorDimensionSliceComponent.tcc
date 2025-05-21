template<typename FloatType, int TensDim>
Tensor<FloatType,TensDim-1> BatchTensorDimensionSliceComponent<FloatType,TensDim>::value(const Tensor<FloatType,TensDim> &in){
  if(!setup){
    memcpy(in_size, in.sizeArray(), TensDim*sizeof(int));
    int dd=0;
    for(int d=0;d<TensDim;d++)
      if(d!=slice_dim)
	out_size[dd++] = in_size[d];
      
    other_dim_vol = 1;
    for(int d=0;d<TensDim-1;d++)
      if(d!=slice_dim)
	other_dim_vol *= in.size(d);

    offset_in = tensorDimensionStride<TensDim>(slice_dim,in_size) * slice_idx;

    setup = true;
  }
  
  Tensor<FloatType,TensDim-1> out(out_size);

  int batch_size = in.size(TensDim-1);
  int _slice_dim = slice_dim;
  size_t _offset_in = offset_in;
  
  {
    autoView(in_v,in,DeviceRead);
    autoView(out_v,out,DeviceWrite);
    
    accelerator_for2d(b,batch_size, o, other_dim_vol, 1, {
	FloatType* in_p = in_v.data() + batchTensorDimensionBaseLin<TensDim>(_slice_dim, b,o, in_v.sizeArray()) + _offset_in;
	FloatType* out_p = out_v.data() + b + batch_size * o;
	*out_p = *in_p;
      });
  }
  
  return out;		  
}

template<typename FloatType, int TensDim>
void BatchTensorDimensionSliceComponent<FloatType,TensDim>::deriv(Tensor<FloatType,TensDim-1> &&_dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const{
  Tensor<FloatType,TensDim-1> dcost_by_dOut(std::move(_dcost_by_dOut));
  dcost_by_dIn = Tensor<FloatType,TensDim>(in_size, 0.);
  int batch_size = in_size[TensDim-1]; 
  int _slice_dim = slice_dim;
  size_t _offset_in = offset_in;
  
  //dCost/dIn_ok = dCost/dOut_o dOut_o/dIn_ojk
  //Out_o = In_ok  delta_{k,slice_idx}
  //dOut_o / dIn_ok = delta_{k,slice_idx}
  //dCost/dIn_ok = dCost/dOut_o delta_{k,slice_idx}
    
  //Out_oj = gamma_j * In_oj + beta_j
  //dOut_oj/dIn_oj = gamma_j
  //dCost/dIn_oj = dCost/dOut_oj gamma_j
    
  autoView(dcost_by_dIn_v, dcost_by_dIn, DeviceReadWrite);
  autoView(dcost_by_dOut_v, dcost_by_dOut, DeviceRead);
    
  accelerator_for2d(b,batch_size, o, other_dim_vol, 1, {
      FloatType* dcost_by_dOut_p = dcost_by_dOut_v.data() + b + batch_size * o;
      FloatType* dcost_by_dIn_p = dcost_by_dIn_v.data() + batchTensorDimensionBaseLin<TensDim>(_slice_dim, b,o, dcost_by_dIn_v.sizeArray() ) + _offset_in;
      *dcost_by_dIn_p = *dcost_by_dOut_p;
    });  
}
