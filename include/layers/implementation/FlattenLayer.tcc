template<typename Config, typename InputType, typename Store>
Matrix<typename Config::FloatType> FlattenLayer<Config,InputType,Store>::value(const InputType &x, EnableDeriv enable_deriv ){
  LayerInputTensorType in = leaf.v.value(x, enable_deriv);
  if(!init){
    memcpy(_input_tens_size, in.sizeArray(), in.dimension() * sizeof(int));
    init = true;
  }
  constexpr int tens_dim = in.dimension();
  _input_tens_size[tens_dim-1] = in.size(tens_dim-1);

  return flattenToBatchVector(in);
}
template<typename Config, typename InputType, typename Store>
int FlattenLayer<Config,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Matrix<FloatType> &&_above_deriv, InputType* input_above_deriv_return) const{
  LayerInputTensorType above_deriv_passdown(_input_tens_size);
  {
    Matrix<FloatType> above_deriv_in(std::move(_above_deriv)); //dcost/dvalue_i
    int batch_size = above_deriv_in.size(1);
    size_t flat_size = above_deriv_in.size(0);

    autoView(above_deriv_passdown_v,above_deriv_passdown,DeviceWrite);
    autoView(above_deriv_in_v,above_deriv_in,DeviceRead);
    accelerator_for2d(b,batch_size, i,flat_size, 1,{
	//rely on the fact that the batch index is the fastest moving
	above_deriv_passdown_v.data()[b + i*batch_size] = above_deriv_in_v(i,b);
      });
  }
  return leaf.v.deriv(cost_deriv,off, std::move(above_deriv_passdown),input_above_deriv_return);
}
template<typename Config, typename InputType, typename Store>
int FlattenLayer<Config,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  return leaf.v.update(off,new_params);
}
template<typename Config, typename InputType, typename Store>   
int FlattenLayer<Config,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  return leaf.v.step(off,derivs,eps);
}
template<typename Config, typename InputType, typename Store>  
int FlattenLayer<Config,InputType,Store>::getParams(Vector<FloatType> &into, int off) const{
  return leaf.v.getParams(into,off);
}
