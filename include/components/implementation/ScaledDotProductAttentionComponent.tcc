template<typename Config>
template<typename InTensorType1, typename InTensorType2, typename InTensorType3, enable_if_fwd_ref3<InTensorType1, InTensorType2,InTensorType3,Tensor<typename Config::FloatType,3> > >
Tensor<typename Config::FloatType,3> ScaledDotProductAttentionComponent<Config>::value(InTensorType1 &&Q, InTensorType2 &&K, InTensorType3 &&V, EnableDeriv enable_deriv){
  //Q(C,d_k,B) ,  K(C,d_k,B)  and V(C,d_v,B)
  assert(Q.size(1) == d_k && K.size(1) == d_k && V.size(1) == d_v);

  if(!setup){
    C = Q.size(0);
    B = Q.size(2);
    setup = true;
  }

  assert(Q.size(0) == C && K.size(0) == C && V.size(0)==C);
  assert(Q.size(2) == B && K.size(2) == B && V.size(2)==B);

  Tensor<FloatType,3> S = mulQKtoGetS.value(std::forward<InTensorType1>(Q),std::forward<InTensorType2>(K), enable_deriv);
  Tensor<FloatType,3> SS = softmaxS_to_SS.value(S, enable_deriv);
  return mulSSVtoGetOut.value(std::move(SS),std::forward<InTensorType3>(V), enable_deriv);
}
template<typename Config>
void ScaledDotProductAttentionComponent<Config>::deriv(Tensor<FloatType,3> &&_dCost_by_dOut, Tensor<FloatType,3> &dCost_by_dQ, Tensor<FloatType,3> &dCost_by_dK, Tensor<FloatType,3> &dCost_by_dV) const{
  Tensor<FloatType,3> dCost_by_dOut = std::move(_dCost_by_dOut);
  assert(dCost_by_dOut.size(0) == C && dCost_by_dOut.size(1) == d_v && dCost_by_dOut.size(2) == B);
  
  Tensor<FloatType,3> dCost_by_dSS;
  mulSSVtoGetOut.deriv(std::move(dCost_by_dOut), dCost_by_dSS, dCost_by_dV);
  
  Tensor<FloatType,3> dCost_by_dS;
  softmaxS_to_SS.deriv(std::move(dCost_by_dSS), dCost_by_dS);
  
  mulQKtoGetS.deriv(std::move(dCost_by_dS), dCost_by_dQ, dCost_by_dK);
}
