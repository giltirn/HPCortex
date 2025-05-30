template<typename FloatType, int TensDim>
Tensor<FloatType,TensDim> MatrixTensorContractComponent<FloatType,TensDim>::value(const Tensor<FloatType,TensDim> &in){
  if(!setup){
    batch_size = in.size(TensDim-1);  
    memcpy(in_dims,in.sizeArray(),TensDim*sizeof(int));
  
    memcpy(out_dims,in.sizeArray(),TensDim*sizeof(int));
    out_dims[TensDim-2] = size0;

    other_size = 1;
    for(int d=0;d<TensDim-2;d++) other_size *= out_dims[d];
    
    setup = true;
  }
  

  if(in.size(TensDim-2) != size1){
    std::stringstream ss; ss << "Expected input features " << in.size(TensDim-2) << " to match number of columns of weight matrix " << size1;
    throw std::runtime_error(ss.str());
  }

  
  Tensor<FloatType,TensDim> out(out_dims); 
  {
    autoView(in_v, in, DeviceRead);
    autoView(weights_v, weights, DeviceRead);
    autoView(out_v, out, DeviceWrite);
    
    int _sizej = size1;
    int _sizei = size0;
    
    //W_{ij} X_{..., j, b}  
    accelerator_for3d(b, batch_size, i, _sizei, o, other_size,    1, {      
	FloatType out_oib = weights_v(i,0) * in_v.compact3(o,0,b);
	for(int j=1;j<_sizej;j++)
	  out_oib += weights_v(i,j) * in_v.compact3(o,j,b);
	out_v.compact3(o,i,b) = out_oib;
      });
  }
   
  in_buf.push(Tensor<FloatType,TensDim>(in)); //TODO: Can avoid this copy in some case by allowing r-value references for inputs. Perhaps have 2 versions of "value", taking l-value and r-value refs, respectively?
  return out;    
}

template<typename FloatType, int TensDim>
void MatrixTensorContractComponent<FloatType,TensDim>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, Tensor<FloatType,TensDim> &layer_deriv) const{
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

  size_t _other_size = other_size;
  int _batch_size = batch_size;
  int _sizei = size0;
  int _sizej = size1;

  //compute layer deriv
  //dcost / dx_oj = \sum_i dcost/df_oi *  w_ij
  layer_deriv = Tensor<FloatType,TensDim>(in_dims);
  {
    autoView(layer_deriv_v,layer_deriv,DeviceWrite);
    autoView(weights_v,weights,DeviceRead);
    autoView(above_deriv_v,above_deriv,DeviceRead);

    accelerator_for3d(b, batch_size, j, _sizej, o, other_size,    1, {
	//dcost / dx_oj = \sum_i dcost/df_oi *  w_ij
	FloatType out_ojb = above_deriv_v.compact3(o,0,b) * weights_v(0,j);
	for(int i=1;i<_sizei;i++)
	  out_ojb += above_deriv_v.compact3(o,i,b) * weights_v(i,j);
	layer_deriv_v.compact3(o,j,b) = out_ojb;	
      });
  }

  //compute cost_deriv
  //dcost / dw_ij = \sum_o dcost/df_oi  x_oj
  {
    autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
    autoView(in_v,in,DeviceRead);
    autoView(above_deriv_v,above_deriv,DeviceRead);

    accelerator_for2d(j,_sizej, i, _sizei,  1, {
	int p = off + j + _sizej*i;
	FloatType cd = 0.;
	for(int o=0;o<_other_size;o++)
	  for(int b=0;b<_batch_size;b++)
	    cd += above_deriv_v.compact3(o,i,b) * in_v.compact3(o,j,b);
	cost_deriv_v(p) = cd;
      });
  }
}

template<typename FloatType, int TensDim>
void MatrixTensorContractComponent<FloatType,TensDim>::update(int off, const Vector<FloatType> &new_params){
  autoView(new_params_v,new_params,DeviceRead);
  autoView(weights_v,weights,DeviceWrite);
  size_t sz1=size1;
  accelerator_for2d(j,size1,i,size0,1,{
      int pp = off + j + sz1*i;
      weights_v(i,j) = new_params_v(pp);
    });
}

template<typename FloatType, int TensDim>
void MatrixTensorContractComponent<FloatType,TensDim>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  autoView(derivs_v,derivs,DeviceRead);
  autoView(weights_v,weights,DeviceReadWrite);
  size_t sz1 = size1;
  accelerator_for2d(j,size1,i,size0,1,{
      int pp = off + j + sz1*i;
      weights_v(i,j) -= derivs_v(pp) * eps;
    });
	  
}


template<typename FloatType, int TensDim>
void MatrixTensorContractComponent<FloatType,TensDim>::getParams(Vector<FloatType> &into, int off){
  autoView(into_v,into,DeviceReadWrite);
  autoView(weights_v,weights,DeviceRead);
  size_t sz1 = size1;
  accelerator_for2d(j,size1,i,size0,1,{
      int pp = off + j + sz1*i;
      into_v(pp) = weights_v(i,j);
    });
}
