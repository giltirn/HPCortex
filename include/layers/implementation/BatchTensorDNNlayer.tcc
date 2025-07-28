template<typename Config, int TensDim, typename InputType, typename Store, typename ActivationFunc>
Tensor<typename Config::FloatType,TensDim> BatchTensorDNNlayer<Config,TensDim,InputType,Store,ActivationFunc>::value(const InputType &x, EnableDeriv enable_deriv){
  return cpt.value(leaf.v.value(x, enable_deriv), enable_deriv);
}

template<typename Config, int TensDim, typename InputType, typename Store, typename ActivationFunc>
int BatchTensorDNNlayer<Config,TensDim,InputType,Store,ActivationFunc>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,TensDim> &&_above_deriv, InputType* input_above_deriv_return) const{
  Tensor<FloatType,TensDim> layer_deriv;
  cpt.deriv(cost_deriv,off,std::move(_above_deriv), layer_deriv);
  return leaf.v.deriv(cost_deriv, off+cpt.nparams(), std::move(layer_deriv), input_above_deriv_return);
}

template<typename Config, int TensDim, typename InputType, typename Store, typename ActivationFunc>
int BatchTensorDNNlayer<Config,TensDim,InputType,Store,ActivationFunc>::update(int off, const Vector<FloatType> &new_params){
  cpt.update(off,new_params);
  return leaf.v.update(off+cpt.nparams(),new_params);
}

template<typename Config, int TensDim, typename InputType, typename Store, typename ActivationFunc>
int BatchTensorDNNlayer<Config,TensDim,InputType,Store,ActivationFunc>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  cpt.step(off,derivs,eps);
  return leaf.v.step(off+cpt.nparams(),derivs,eps);
}

template<typename Config, int TensDim, typename InputType, typename Store, typename ActivationFunc>
int BatchTensorDNNlayer<Config,TensDim,InputType,Store,ActivationFunc>::getParams(Vector<FloatType> &into, int off) const{
  cpt.getParams(into,off);
  return leaf.v.getParams(into,off+cpt.nparams());
}
