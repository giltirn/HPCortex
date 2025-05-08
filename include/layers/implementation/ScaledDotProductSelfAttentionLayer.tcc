template<typename FloatType, typename InputType, typename Store>
Tensor<FloatType,3> ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::value(const InputType &x){
  Tensor<FloatType,3> X = leaf.v.value(x);
  assert(X.size(1) == E);
  if(!setup){
    C = X.size(0);
    B = X.size(2);
    setup = true;
  }else assert(X.size(0) == C && X.size(2) == B);

  Tensor<FloatType,3> Q = multWQ.value(X);
  Tensor<FloatType,3> K = multWK.value(X);
  Tensor<FloatType,3> V = multWV.value(X);
  return attentionQKV.value(Q,K,V);
}

template<typename FloatType, typename InputType, typename Store>
void ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_above_deriv, InputType* input_above_deriv_return) const{
  //Params are in order of W_Q, W_K, W_V
  Tensor<FloatType,3> layer_deriv(C,E,B);
  int p = off;
  {
    Tensor<FloatType,3> above_deriv_Out = std::move(_above_deriv);
    assert(above_deriv_Out.size(0) == C && above_deriv_Out.size(1) == d_v && above_deriv_Out.size(2) == B);
    
    Tensor<FloatType,3> above_deriv_Q, above_deriv_K, above_deriv_V;
    attentionQKV.deriv(std::move(above_deriv_Out), above_deriv_Q, above_deriv_K, above_deriv_V);
    
    Tensor<FloatType,3> layer_deriv_K, layer_deriv_Q, layer_deriv_V; //these are  dCost/dX_{ceb} through the different routes
    multWQ.deriv(cost_deriv, p, std::move(above_deriv_Q), layer_deriv_Q);
    p += multWQ.nparams();
    multWK.deriv(cost_deriv, p, std::move(above_deriv_K), layer_deriv_K);
    p += multWK.nparams();
    multWV.deriv(cost_deriv, p, std::move(above_deriv_V), layer_deriv_V);
    p += multWV.nparams();

    //Sum dCost/dX_{ceb} through the different routes
    {
      autoView(layer_deriv_K_v,layer_deriv_K,DeviceRead);
      autoView(layer_deriv_Q_v,layer_deriv_Q,DeviceRead);
      autoView(layer_deriv_V_v,layer_deriv_V,DeviceRead);
      autoView(layer_deriv_v,layer_deriv,DeviceWrite);
      accelerator_for3d(b,B,e,E,c,C, 1, {
	  layer_deriv_v(c,e,b) = layer_deriv_K_v(c,e,b) + layer_deriv_Q_v(c,e,b) + layer_deriv_V_v(c,e,b);
	});
    }
  }
  leaf.v.deriv(cost_deriv, p, std::move(layer_deriv),  input_above_deriv_return);
}

template<typename FloatType, typename InputType, typename Store>
void ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::update(int off, const Vector<FloatType> &new_params){
  int p = off;
  multWQ.update(p,new_params);
  p += multWQ.nparams();
  multWK.update(p,new_params);
  p += multWK.nparams();
  multWV.update(p,new_params);
  p += multWV.nparams();
  leaf.v.update(p,new_params);
}
  
template<typename FloatType, typename InputType, typename Store>
void ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  int p = off;
  multWQ.step(p,derivs,eps);
  p += multWQ.nparams();
  multWK.step(p,derivs,eps);
  p += multWK.nparams();
  multWV.step(p,derivs,eps);
  p += multWV.nparams();
  leaf.v.step(p,derivs,eps);
}

//off measured from *end*, return new off
template<typename FloatType, typename InputType, typename Store>
void ScaledDotProductSelfAttentionLayer<FloatType,InputType,Store>::getParams(Vector<FloatType> &into, int off){
  int p = off;
  multWQ.getParams(into,p);
  p += multWQ.nparams();
  multWK.getParams(into,p);
  p += multWK.nparams();
  multWV.getParams(into,p);
  p += multWV.nparams();
  leaf.v.getParams(into,p);
}
