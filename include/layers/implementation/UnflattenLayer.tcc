template<typename Config, int OutDimension, typename InputType, typename Store>
Tensor<typename Config::FloatType,OutDimension> UnflattenLayer<Config,OutDimension,InputType,Store>::value(const InputType &x, EnableDeriv enable_deriv){
  LayerInputTensorType in = leaf.v.value(x, enable_deriv);
  int batch_size = in.size(1);
  _output_tens_size[OutDimension-1] = batch_size;
  
  size_t flat_size = 1;
  for(int i=0;i<OutDimension-1;i++)
    flat_size *= _output_tens_size[i];

  if(in.size(0) != flat_size){
    std::ostringstream ss; ss << "Expected input matrix first dimension size " << flat_size << ", got " << in.size(0);    
    throw std::runtime_error(ss.str());
  }

  Tensor<FloatType,OutDimension> out(_output_tens_size);
  autoView(out_v,out,DeviceWrite);
  autoView(in_v,in,DeviceRead);
  accelerator_for2d(b,batch_size, i,flat_size, 1,{
      //rely on the fact that the batch index is the fastest moving,  eg. for a 3 tensor   off = b + batch_size*(z + zsize*(y + ysize*x))      i=(z + zsize*(y + ysize*x)) 
      out_v.data()[b + batch_size*i] = in_v(i,b);
    });
  return out;
}
template<typename Config, int OutDimension, typename InputType, typename Store>
int UnflattenLayer<Config,OutDimension,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,OutDimension> &&_above_deriv, InputType* input_above_deriv_return) const{
  int batch_size = _output_tens_size[OutDimension-1];
  size_t flat_size = 1;
  for(int i=0;i<OutDimension-1;i++)
    flat_size *= _output_tens_size[i];
  
  Matrix<FloatType> above_deriv_passdown(flat_size,batch_size);
  {
    Tensor<FloatType,OutDimension> above_deriv_in(std::move(_above_deriv)); //dcost/dvalue_i
    for(int d=0;d<OutDimension;d++) assert(above_deriv_in.size(d) == _output_tens_size[d]);

    autoView(above_deriv_passdown_v,above_deriv_passdown,DeviceWrite);
    autoView(above_deriv_in_v,above_deriv_in,DeviceRead);
    accelerator_for2d(b,batch_size, i,flat_size, 1,{
	//rely on the fact that the batch index is the fastest moving
	above_deriv_passdown_v(i,b) = above_deriv_in_v.data()[b + i*batch_size];
      });
  }
  return leaf.v.deriv(cost_deriv,off, std::move(above_deriv_passdown),input_above_deriv_return);
}
template<typename Config, int OutDimension, typename InputType, typename Store>
int UnflattenLayer<Config,OutDimension,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  return leaf.v.update(off,new_params);
}
template<typename Config, int OutDimension, typename InputType, typename Store>   
int UnflattenLayer<Config,OutDimension,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  return leaf.v.step(off,derivs,eps);
}
template<typename Config, int OutDimension, typename InputType, typename Store>  
int UnflattenLayer<Config,OutDimension,InputType,Store>::getParams(Vector<FloatType> &into, int off) const{
  return leaf.v.getParams(into,off);
}
