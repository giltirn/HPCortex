template<typename FloatType, int TensDim, typename InputType, typename Store>
Tensor<FloatType,TensDim> MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::value(const InputType &x){
  return cpt.value(leaf.v.value(x));

  // if(!setup){
  //   batch_size = in.size(TensDim-1);  
  //   memcpy(in_dims,in.sizeArray(),TensDim*sizeof(int));
  
  //   memcpy(out_dims,in.sizeArray(),TensDim*sizeof(int));
  //   out_dims[TensDim-2] = size0;

  //   other_size = 1;
  //   for(int d=0;d<TensDim-2;d++) other_size *= out_dims[d];
    
  //   setup = true;
  // }
  

  // if(in.size(TensDim-2) != size1){
  //   std::stringstream ss; ss << "Expected input features " << in.size(TensDim-2) << " to match number of columns of weight matrix " << size1;
  //   throw std::runtime_error(ss.str());
  // }

  
  // Tensor<FloatType,TensDim> out(out_dims); 
  // {
  //   autoView(in_v, in, DeviceRead);
  //   autoView(weights_v, weights, DeviceRead);
  //   autoView(out_v, out, DeviceWrite);
    
  //   int _batch_size = batch_size;
  //   int _sizej = size1;
  //   int _sizei = size0;
    
  //   //W_{ij} X_{..., j, b}  
  //   accelerator_for3d(b, batch_size, i, _sizei, o, other_size,    1, {      
  // 	FloatType out_oib = weights_v(i,0) * in_v.compact3(o,0,b);
  // 	for(int j=1;j<_sizej;j++)
  // 	  out_oib += weights_v(i,j) * in_v.compact3(o,j,b);
  // 	out_v.compact3(o,i,b) = out_oib;
  //     });
  // }
   
  // leaf_buf.push(std::move(in));
  // return out;    
}

template<typename FloatType, int TensDim, typename InputType, typename Store>
void MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, InputType* input_above_deriv_return) const{
  Tensor<FloatType,TensDim> layer_deriv;
  cpt.deriv(cost_deriv,off,std::move(_above_deriv), layer_deriv);
  
  // int p = off;
  // assert(_above_deriv.size(TensDim-2) == size0);
  // assert(_above_deriv.size(TensDim-1) == batch_size);
  
  // //f_oi = \sum_j w_ij x_oj      for compound index o
  // //dcost / dx_oj = \sum_i dcost/df_oi *  df_oi / dx_oj   : "layer_deriv" 
  // //df_oi / dx_oj = w_ij 
  // //dcost / dx_oj = \sum_i dcost/df_oi *  w_ij

  // //dcost / dw_ij = \sum_i'o   dcost/df_oi'  *  df_oi' / dw_ij 
  // //df_oi' / dw_ij = x_oj \delta_i'i
  // //dcost / dw_ij = \sum_o dcost/df_oi  x_oj
  
    
  // Tensor<FloatType,TensDim> layer_deriv;
  // {
  //   Tensor<FloatType,TensDim> above_deriv(std::move(_above_deriv)); //inside the braces above ensures this object is freed before the next layer is called
      
  //   Tensor<FloatType,TensDim> in = leaf_buf.isFilled() ? leaf_buf.pop(): Tensor<FloatType,TensDim>(in_dims,0.);
  //   assert(in.size(TensDim-2) == size1);
  //   assert(in.size(TensDim-1) == batch_size);

  //   size_t _other_size = other_size;
  //   int _batch_size = batch_size;
  //   int _sizei = size0;
  //   int _sizej = size1;

  //   //compute layer deriv
  //   //dcost / dx_oj = \sum_i dcost/df_oi *  w_ij
  //   layer_deriv = Tensor<FloatType,TensDim>(in_dims);
  //   {
  //     autoView(layer_deriv_v,layer_deriv,DeviceWrite);
  //     autoView(weights_v,weights,DeviceRead);
  //     autoView(above_deriv_v,above_deriv,DeviceRead);
      
  //     accelerator_for3d(b, batch_size, j, _sizej, o, other_size,    1, {
  // 	//dcost / dx_oj = \sum_i dcost/df_oi *  w_ij
  // 	FloatType out_ojb = above_deriv_v.compact3(o,0,b) * weights_v(0,j);
  // 	for(int i=1;i<_sizei;i++)
  // 	  out_ojb += above_deriv_v.compact3(o,i,b) * weights_v(i,j);
  // 	layer_deriv_v.compact3(o,j,b) = out_ojb;	
  // 	});
  //   }

  //   //compute cost_deriv
  //   //dcost / dw_ij = \sum_o dcost/df_oi  x_oj
  //   {
  //     autoView(cost_deriv_v,cost_deriv,DeviceReadWrite);
  //     autoView(in_v,in,DeviceRead);
  //     autoView(above_deriv_v,above_deriv,DeviceRead);
      
  //     accelerator_for2d(j,_sizej, i, _sizei,  1, {
  // 	  int p = off + j + _sizej*i;
  // 	  FloatType cd = 0.;
  // 	  for(int o=0;o<_other_size;o++)
  // 	    for(int b=0;b<_batch_size;b++)
  // 	      cd += above_deriv_v.compact3(o,i,b) * in_v.compact3(o,j,b);
  // 	  cost_deriv_v(p) = cd;
  // 	});

  //     p+=size0*size1;
  //   }
  // }//close views and free temporaries before calling layer below
    
  // leaf.v.deriv(cost_deriv, p, std::move(layer_deriv), input_above_deriv_return);

  leaf.v.deriv(cost_deriv, off+cpt.nparams(), std::move(layer_deriv), input_above_deriv_return);
}

template<typename FloatType, int TensDim, typename InputType, typename Store>
void MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  cpt.update(off,new_params);
  leaf.v.update(off+cpt.nparams(),new_params);
  
  
  // int p = off;
  // {
  //   autoView(new_params_v,new_params,DeviceRead);
  //   autoView(weights_v,weights,DeviceWrite);
  //   size_t sz1=size1;
  //   accelerator_for2d(j,size1,i,size0,1,{
  // 	int pp = p + j + sz1*i;
  // 	weights_v(i,j) = new_params_v(pp);
  //     });
	  
  //   p += size0*size1;
  // }
  // leaf.v.update(p, new_params);
}

template<typename FloatType, int TensDim, typename InputType, typename Store>
void MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  cpt.step(off,derivs,eps);
  leaf.v.step(off+cpt.nparams(),derivs,eps);
  
  // int p=off;
  // {
  //   autoView(derivs_v,derivs,DeviceRead);
  //   autoView(weights_v,weights,DeviceReadWrite);
  //   size_t sz1 = size1;
  //   accelerator_for2d(j,size1,i,size0,1,{
  // 	int pp = p + j + sz1*i;
  // 	weights_v(i,j) -= derivs_v(pp) * eps;
  //     });
	  
  //   p += size0*size1;
  // }
  // leaf.v.step(p, derivs, eps);
}



//off measured from *end*, return new off
template<typename FloatType, int TensDim, typename InputType, typename Store>
void MatrixTensorContractLayer<FloatType,TensDim,InputType,Store>::getParams(Vector<FloatType> &into, int off){
  cpt.getParams(into,off);
  leaf.v.getParams(into,off+cpt.nparams());
  
  // int p = off;
  // {
  //   autoView(into_v,into,DeviceReadWrite);
  //   autoView(weights_v,weights,DeviceRead);
  //   size_t sz1 = size1;
  //   accelerator_for2d(j,size1,i,size0,1,{
  // 	int pp = p + j + sz1*i;
  // 	into_v(pp) = weights_v(i,j);
  //     });

  //   p += size0*size1;
  // }
  // leaf.v.getParams(into, p);
}
