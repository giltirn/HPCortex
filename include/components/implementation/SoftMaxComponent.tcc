template<typename Config, int TensDim>
Tensor<typename Config::FloatType,TensDim> SoftMaxComponent<Config,TensDim>::value(const Tensor<FloatType,TensDim> &in) const{
  int batch_size = in.size(TensDim-1);
  int nlogp = in.size(softmax_dim);

  FloatType beta_ = beta;

  Tensor<FloatType,TensDim> out(in.sizeArray());
  
  int softmax_dim_ = softmax_dim;
  
  size_t other_dim_vol = 1;
  for(int d=0;d<TensDim-1;d++)
    if(d!=softmax_dim)
      other_dim_vol *= in.size(d);

  size_t stride = tensorDimensionStride<TensDim>(softmax_dim,in.sizeArray());

  if(!value_FLOPS.locked()){ //note, this count ignores the recomputation of the norm when it hits a new max value
    value_FLOPS.add(other_dim_vol*batch_size* 8*nlogp);    
    value_FLOPS.lock();
  }
  
  {
    autoView(in_v,in,DeviceRead);
    autoView(out_v,out,DeviceWrite);
  
    accelerator_for2d(b,batch_size, o, other_dim_vol, 1, {
	size_t off = batchTensorDimensionBaseLin<TensDim>(softmax_dim_, b,o, in_v.sizeArray());      
	FloatType* in_p = in_v.data() + off;
      
	FloatType max = *in_p; in_p += stride;
	FloatType norm = 1.0;   //exp(beta_*(in_v(0,b)-max));

	for(int i=1;i<nlogp;i++){
	  FloatType ii = *in_p; in_p += stride;

	  if(ii > max){
	    FloatType old_max = max;
	    max = ii;
	    norm = norm * exp(beta_*(old_max-max) ) + 1.0;
	  }else{	
	    norm += exp(beta_*(ii-max));
	  }	  
	}

	in_p = in_v.data() + off;
	FloatType* out_p = out_v.data() + off;
      
	for(int i=0;i<nlogp;i++){
	  *out_p = exp(beta_*( (*in_p)-max)) / norm;
	  out_p += stride;
	  in_p += stride;
	}
      });
  }
  
  out_buf.push(Tensor<FloatType,TensDim>(out));
  
  return out;
}

template<typename Config, int TensDim>
void SoftMaxComponent<Config,TensDim>::deriv(Tensor<FloatType,TensDim> &&_dcost_by_dOut, Tensor<FloatType,TensDim> &dcost_by_dIn) const{
  //No parameters so we just have to compute the "layer_deriv",  l_j = \sum_i dcost/dout_i dout_i / din_j
  //out_i =  exp(beta*in_i)/ \sum_k exp(beta*in_k) = e_i / norm
  //dout_i / din_j = beta * out_i \delta_ij     - e_i  beta e_j /norm^2 = beta * out_i ( \delta_ij - out_j )
  ///                                                                  
  //dcost/din_j = beta * \sum_i dcost/dout_i out_i ( \delta_ij - out_j )
  //            = beta * dcost/dout_j out_j  - beta * out_j * \sum_i dcost/dout_i out_i 
  //            = beta * out_j * ( dcost/dout_j  - \sum_i dcost/dout_i out_i )
  Tensor<FloatType,TensDim> out = out_buf.isFilled() ? out_buf.pop(): out_buf.latest();
  Tensor<FloatType,TensDim> dcost_by_dOut(std::move(_dcost_by_dOut)); //take ownership so destroyed at scope close
    
  dcost_by_dIn = Tensor<FloatType,TensDim>(out.sizeArray());
  int nlogp = out.size(softmax_dim);
  FloatType beta_ = beta;
  int softmax_dim_ = softmax_dim;
  int batch_size = out.size(TensDim-1);

  size_t other_dim_vol = 1;
  for(int d=0;d<TensDim-1;d++)
    if(d!=softmax_dim)
      other_dim_vol *= out.size(d);

  size_t stride = tensorDimensionStride<TensDim>(softmax_dim,out.sizeArray());

  if(!deriv_FLOPS.locked()){ 
    deriv_FLOPS.add(other_dim_vol*nlogp*batch_size* (2*nlogp + 2) );    
    deriv_FLOPS.lock();
  }
  
  autoView(out_v,out,DeviceRead);
  autoView(dcost_by_dOut_v, dcost_by_dOut, DeviceRead);
  autoView(dcost_by_dIn_v, dcost_by_dIn, DeviceWrite);
  
  accelerator_for3d(b, batch_size, j, nlogp, o, other_dim_vol, 1, {
      size_t off = batchTensorDimensionBaseLin<TensDim>(softmax_dim_, b,o, out_v.sizeArray());
      size_t joff = off + j*stride;
      FloatType lj =  dcost_by_dOut_v.data()[joff];

      FloatType *dcost_by_dOut_p = dcost_by_dOut_v.data() + off;
      FloatType *out_p = out_v.data() + off;
	
      for(int i=0;i<nlogp;i++){
	lj -= (*dcost_by_dOut_p) * (*out_p);
	dcost_by_dOut_p += stride;
	out_p += stride;
      }
	
      dcost_by_dIn_v.data()[joff] = beta_ * out_v.data()[joff] * lj;
    });
}

