template<typename FloatType, int TensDim, typename InputType, typename Store >
Tensor<FloatType,TensDim> SoftMaxLayer<FloatType,TensDim,InputType,Store>::value(const InputType &x){	
  return cpt.value(leaf.v.value(x));
}

template<typename FloatType, int TensDim, typename InputType, typename Store >
int SoftMaxLayer<FloatType,TensDim,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&above_deriv_, InputType* input_above_deriv_return) const{
  Tensor<FloatType,TensDim> layer_deriv;
  cpt.deriv(std::move(above_deriv_), layer_deriv);
  return leaf.v.deriv(cost_deriv,off,std::move(layer_deriv),input_above_deriv_return);        
}
