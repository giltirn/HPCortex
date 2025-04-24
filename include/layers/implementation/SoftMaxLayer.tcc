template<typename FloatType, typename InputType, typename Store >
Matrix<FloatType> SoftMaxLayer<FloatType,InputType,Store>::value(const InputType &x){	
  Matrix<FloatType> in = leaf.v.value(x);
  batch_size = in.size(1);
  nlogp = in.size(0);

  FloatType beta_ = beta;
  int nlogp_ = nlogp;
  
  Matrix<FloatType> out(nlogp,batch_size);
  
  autoView(in_v,in,DeviceRead);
  autoView(out_v,out,DeviceWrite);
  
  accelerator_for(b,batch_size, {
	// FloatType max = in_v(0,b);
	// for(int i=1;i<nlogp;i++)
	//   max = in_v(i,b) > max ? in_v(i,b) : max;

	// FloatType norm = exp(beta_*(in_v(0,b)-max));
	// for(int i=1;i<nlogp;i++)
	//   norm += exp(beta_*(in_v(i,b)-max));
	
	// for(int i=0;i<nlogp;i++){
	//   out_v(i,b) = exp(beta_*(in_v(i,b)-max)) / norm;
	// }


      
      FloatType max = in_v(0,b);
      FloatType norm = 1.0;   //exp(beta_*(in_v(0,b)-max));

      for(int i=1;i<nlogp_;i++){
	FloatType ii = in_v(i,b);

	if(ii > max){
	  FloatType old_max = max;
	  max = ii;
	  norm = norm * exp(beta_*(old_max-max) ) + 1.0;
	}else{	
	  norm += exp(beta_*(ii-max));
	}
	  
      }
      for(int i=0;i<nlogp_;i++)
	out_v(i,b) = exp(beta_*(in_v(i,b)-max)) / norm;
    });

  out_buf.push(Matrix<FloatType>(out));
  
  return out;
}

template<typename FloatType, typename InputType, typename Store >
void SoftMaxLayer<FloatType,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&above_deriv_, InputType* input_above_deriv_return) const{
  Matrix<FloatType> layer_deriv;
  {
    //No parameters so we just have to compute the "layer_deriv",  l_j = \sum_i dcost/dout_i dout_i / din_j
    //out_i =  exp(beta*in_i)/ \sum_k exp(beta*in_k) = e_i / norm
    //dout_i / din_j = beta * out_i \delta_ij     - e_i  beta e_j /norm^2 = beta * out_i ( \delta_ij - out_j )
    ///                                                                  
    //dcost/din_j = beta * \sum_i dcost/dout_i out_i ( \delta_ij - out_j )
    //            = beta * dcost/dout_j out_j  - beta * out_j * \sum_i dcost/dout_i out_i 
    //            = beta * out_j * ( dcost/dout_j  - \sum_i dcost/dout_i out_i )
    Matrix<FloatType> out = out_buf.isFilled() ? out_buf.pop(): Matrix<FloatType>(nlogp,batch_size,0.);
    Matrix<FloatType> above_deriv(std::move(above_deriv_)); //take ownership so destroyed at scope close
    
    layer_deriv = Matrix<FloatType>(nlogp,batch_size);

    int nlogp_ = nlogp;
    FloatType beta_ = beta;

    autoView(out_v,out,DeviceRead); //(i, b)
    autoView(above_deriv_v, above_deriv, DeviceRead); //(i, b)
    autoView(layer_deriv_v, layer_deriv, DeviceWrite);
    
    accelerator_for2d(b, batch_size, j, nlogp, 1, {
	FloatType lj =  above_deriv_v(j,b);
	
	for(int i=0;i<nlogp_;i++)
	  lj -= above_deriv_v(i,b) * out_v(i,b);

	layer_deriv_v(j,b) = beta_ * out_v(j,b) * lj;
      });
  }
  leaf.v.deriv(cost_deriv,off,std::move(layer_deriv),input_above_deriv_return);        
}
