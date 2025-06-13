template<typename FloatType>
Tensor<FloatType,3> ScaledDotProductAttentionHeadComponent<FloatType>::value(const Tensor<FloatType,3> &Q, const Tensor<FloatType,3> &K, const Tensor<FloatType,3> &V){
  //Q(C,E,B) ,  K(C,E,B)  and V(C,E,B)
  assert(Q.size(1) == E && K.size(1) == E && V.size(1) == E);

  if(!setup){
    C = Q.size(0);
    B = Q.size(2);
    setup = true;
  }

  assert(Q.size(0) == C && K.size(0) == C && V.size(0)==C);
  assert(Q.size(2) == B && K.size(2) == B && V.size(2)==B);

  Tensor<FloatType,3> Qp = multWQ.value(Q);
  Tensor<FloatType,3> Kp = multWK.value(K);
  Tensor<FloatType,3> Vp = multWV.value(V);
  return attention.value(Qp,Kp,Vp);
}
template<typename FloatType>
void ScaledDotProductAttentionHeadComponent<FloatType>::deriv(Vector<FloatType> &cost_deriv, int off, Tensor<FloatType,3> &&_dCost_by_dOut, Tensor<FloatType,3> &dCost_by_dQ, Tensor<FloatType,3> &dCost_by_dK, Tensor<FloatType,3> &dCost_by_dV) const{ 
  Tensor<FloatType,3> dCost_by_dOut = std::move(_dCost_by_dOut);
  assert(dCost_by_dOut.size(0) == C && dCost_by_dOut.size(1) == d_v && dCost_by_dOut.size(2) == B);

  //Params are in order of W_Q, W_K, W_V
  int p = off;
    
  Tensor<FloatType,3> above_deriv_Qp, above_deriv_Kp, above_deriv_Vp;
  attention.deriv(std::move(dCost_by_dOut), above_deriv_Qp, above_deriv_Kp, above_deriv_Vp);
    
  Tensor<FloatType,3> layer_deriv_K, layer_deriv_Q, layer_deriv_V; //these are  dCost/dX_{ceb} through the different routes
  multWQ.deriv(cost_deriv, p, std::move(above_deriv_Qp), dCost_by_dQ);
  p += multWQ.nparams();
  multWK.deriv(cost_deriv, p, std::move(above_deriv_Kp), dCost_by_dK);
  p += multWK.nparams();
  multWV.deriv(cost_deriv, p, std::move(above_deriv_Vp), dCost_by_dV);
}

template<typename FloatType>
void ScaledDotProductAttentionHeadComponent<FloatType>::update(int off, const Vector<FloatType> &new_params){
  int p = off;
  multWQ.update(p,new_params);
  p += multWQ.nparams();
  multWK.update(p,new_params);
  p += multWK.nparams();
  multWV.update(p,new_params);
}
  
template<typename FloatType>
void ScaledDotProductAttentionHeadComponent<FloatType>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  int p = off;
  multWQ.step(p,derivs,eps);
  p += multWQ.nparams();
  multWK.step(p,derivs,eps);
  p += multWK.nparams();
  multWV.step(p,derivs,eps);
}

//off measured from *end*, return new off
template<typename FloatType>
void ScaledDotProductAttentionHeadComponent<FloatType>::getParams(Vector<FloatType> &into, int off) const{
  int p = off;
  multWQ.getParams(into,p);
  p += multWQ.nparams();
  multWK.getParams(into,p);
  p += multWK.nparams();
  multWV.getParams(into,p);
}
