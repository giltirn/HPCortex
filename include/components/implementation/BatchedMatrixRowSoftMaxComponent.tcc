template<typename Config>
Tensor<typename Config::FloatType,3> BatchedMatrixRowSoftMaxComponent<Config>::value(const Tensor<FloatType,3> &in, EnableDeriv enable_deriv) const{
  int batch_size = in.size(2);
  int rows = in.size(0);
  int cols = in.size(1);
  if(use_mask) assert(rows == cols);

  FloatType beta_ = beta;

  Tensor<FloatType,3> out(in.sizeArray(),0.);
  bool use_mask_ = use_mask;

  if(!value_FLOPS.locked()){ //note, this count ignores the recomputation of the norm when it hits a new max value
    size_t cnt = 0;
    for(int r=0;r<rows;r++){
      int nlogp = use_mask ? r+1 : cols;
      cnt += 8*nlogp;
    }
    value_FLOPS.add(cnt*batch_size);
    value_FLOPS.lock();
  }   
  
  {
    autoView(in_v,in,DeviceRead);
    autoView(out_v,out,DeviceReadWrite);
  
    accelerator_for2d(b,batch_size, r, rows, 1, {
	int nlogp = use_mask_ ? r+1 : cols;  //masking adds -inf to all column elements > row index of the input, so we only need to factor in elements c<=r
	FloatType max = in_v(r,0,b);
	FloatType norm = 1.0;   //exp(beta_*(in_v(0,b)-max));
	  
	for(int i=1;i<nlogp;i++){
	  FloatType ii = in_v(r,i,b);

	  if(ii > max){
	    FloatType old_max = max;
	    max = ii;
	    norm = norm * exp(beta_*(old_max-max) ) + 1.0;
	  }else{	
	    norm += exp(beta_*(ii-max));
	  }	  
	}

	for(int i=0;i<nlogp;i++)
	  out_v(r,i,b) = exp(beta_*( in_v(r,i,b)-max)) / norm;
	      
      });
  }
  
  if(enable_deriv) out_buf.push(Tensor<FloatType,3>(out));
  
  return out;
}

template<typename Config>
void BatchedMatrixRowSoftMaxComponent<Config>::deriv(Tensor<FloatType,3> &&_dcost_by_dOut, Tensor<FloatType,3> &dcost_by_dIn) const{
  //No parameters so we just have to compute the "layer_deriv",  l_j = \sum_i dcost/dout_i dout_i / din_j
  //out_i =  exp(beta*in_i)/ \sum_k exp(beta*in_k) = e_i / norm
  //dout_i / din_j = beta * out_i \delta_ij     - e_i  beta e_j /norm^2 = beta * out_i ( \delta_ij - out_j )
  ///                                                                  
  //dcost/din_j = beta * \sum_i dcost/dout_i out_i ( \delta_ij - out_j )
  //            = beta * dcost/dout_j out_j  - beta * out_j * \sum_i dcost/dout_i out_i 
  //            = beta * out_j * ( dcost/dout_j  - \sum_i dcost/dout_i out_i )
  Tensor<FloatType,3> out = out_buf.isFilled() ? out_buf.pop(): out_buf.latest();
  Tensor<FloatType,3> dcost_by_dOut(std::move(_dcost_by_dOut)); //take ownership so destroyed at scope close
    
  dcost_by_dIn = Tensor<FloatType,3>(out.sizeArray());
  int rows = out.size(0);
  int cols = out.size(1);
  int batch_size = out.size(2);
  
  FloatType beta_ = beta;
  bool use_mask_ = use_mask;

  if(!deriv_FLOPS.locked()){
    size_t cnt = 0;
    for(int r=0;r<rows;r++){
      int nlogp = use_mask ? r+1 : cols;
      cnt += nlogp * (2*nlogp + 2);
    }
    deriv_FLOPS.add(cnt * batch_size);
    deriv_FLOPS.lock();
  }   
  
  autoView(out_v,out,DeviceRead);
  autoView(dcost_by_dOut_v, dcost_by_dOut, DeviceRead);
  autoView(dcost_by_dIn_v, dcost_by_dIn, DeviceWrite);
  
  accelerator_for3d(b, batch_size, j, cols, r, rows, 1, {
      if(use_mask_ && j>r) dcost_by_dIn_v(r,j,b) = 0.;
      else{
	int nlogp = use_mask_ ? r+1 : cols;	
	FloatType lj =  dcost_by_dOut_v(r,j,b);
	
	for(int i=0;i<nlogp;i++)
	  lj -= dcost_by_dOut_v(r,i,b) * out_v(r,i,b);
	
	dcost_by_dIn_v(r,j,b) = beta_ * out_v(r,j,b) * lj;
      }
    });
}

