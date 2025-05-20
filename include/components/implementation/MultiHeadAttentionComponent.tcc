template<typename FloatType>
MultiHeadAttentionComponent<FloatType>::MultiHeadAttentionComponent(int Nheads, Matrix<FloatType> const* const* W_Q, Matrix<FloatType> const* const* W_K, Matrix<FloatType> const* const* W_V, const Matrix<FloatType> &W_O, bool use_mask):
  concatY(1,Nheads),
  multW_O(W_O),
  heads(Nheads),
  setup(false)
{
  assert(Nheads > 0);
  //ensure that d_v all sum to the right size
  int sdv=0;
  for(int h=0;h<Nheads;h++)
    sdv += W_V[h]->size(0);
  assert(sdv == W_O.size(1));

  E = W_K[0]->size(1);
  for(int h=1;h<Nheads;h++)
    assert(W_K[h]->size(1) == E);

  Nparams_layer = multW_O.nparams();
  for(int h=0;h<Nheads;h++){
    heads[h].reset(new ScaledDotProductAttentionHeadComponent<FloatType>(*W_Q[h],*W_K[h],*W_V[h],use_mask));
    Nparams_layer += heads[h]->nparams();
  }

}

template<typename FloatType>
MultiHeadAttentionComponent<FloatType>::MultiHeadAttentionComponent(int Nheads, const std::vector<Matrix<FloatType> > &W_Q, const std::vector<Matrix<FloatType> > &W_K, const std::vector<Matrix<FloatType> > &W_V, const Matrix<FloatType> &W_O, bool use_mask):
  concatY(1,Nheads),
  multW_O(W_O),
  heads(Nheads),
  setup(false)
{
  assert(Nheads > 0);
  //ensure that d_v all sum to the right size
  int sdv=0;
  for(int h=0;h<Nheads;h++)
    sdv += W_V[h].size(0);
  assert(sdv == W_O.size(1));

  E = W_K[0].size(1);
  for(int h=1;h<Nheads;h++)
    assert(W_K[h].size(1) == E);

  Nparams_layer = multW_O.nparams();
  for(int h=0;h<Nheads;h++){
    heads[h].reset(new ScaledDotProductAttentionHeadComponent<FloatType>(W_Q[h],W_K[h],W_V[h],use_mask));
    Nparams_layer += heads[h]->nparams();
  }

}




template<typename FloatType>
Tensor<FloatType,3> MultiHeadAttentionComponent<FloatType>::value(const TensorType &Q, const TensorType &K, const TensorType &V){
  //Q(C,E,B) ,  K(C,E,B)  and V(C,E,B)
  assert(Q.size(1) == E && K.size(1) == E && V.size(1) == E);

  if(!setup){
    C = Q.size(0);
    B = Q.size(2);
    setup = true;
  }

  assert(Q.size(0) == C && K.size(0) == C && V.size(0)==C);
  assert(Q.size(2) == B && K.size(2) == B && V.size(2)==B);

  std::vector< TensorType > Y(heads.size());
  std::vector< TensorType const* > Yp(heads.size());
  for(int h=0;h<heads.size();h++){  
    Y[h] = heads[h]->value(Q,K,V);
    Yp[h] = &Y[h];
  }
  return multW_O.value( concatY.value(Yp.data()) );
}
template<typename FloatType>
void MultiHeadAttentionComponent<FloatType>::deriv(Vector<FloatType> &cost_deriv, int off, TensorType &&dCost_by_dOut, TensorType &dCost_by_dQ, TensorType &dCost_by_dK, TensorType &dCost_by_dV) const{
  int p = off;
  
  TensorType dCost_by_dO = std::move(dCost_by_dOut);
  TensorType dCost_by_dYconcat;
  multW_O.deriv(cost_deriv, p, std::move(dCost_by_dO), dCost_by_dYconcat);
  p += multW_O.nparams();
    
  std::vector< TensorType > dcost_by_dY(heads.size());
  std::vector< TensorType* > dcost_by_dYp(heads.size());
  for(int h=0;h<heads.size();h++) dcost_by_dYp[h] = &dcost_by_dY[h];
  concatY.deriv(std::move(dCost_by_dYconcat), dcost_by_dYp.data());

  dCost_by_dQ = TensorType(C,E,B, 0.);
  dCost_by_dK = TensorType(C,E,B, 0.);
  dCost_by_dV = TensorType(C,E,B, 0.);
  
  for(int h=0;h<heads.size();h++){
    TensorType dCost_by_dX_Q, dCost_by_dX_K, dCost_by_dX_V;
    heads[h]->deriv(cost_deriv, p, std::move(dcost_by_dY[h]), dCost_by_dX_Q, dCost_by_dX_K, dCost_by_dX_V);
    p += heads[h]->nparams();

    TensorType* to[3] = { &dCost_by_dQ, &dCost_by_dK, &dCost_by_dV };
    TensorType* from[3] = { &dCost_by_dX_Q, &dCost_by_dX_K, &dCost_by_dX_V };
    for(int i=0;i<3;i++){    
      autoView(to_v, (*to[i]), DeviceReadWrite);
      autoView(from_v, (*from[i]), DeviceRead);
      
      accelerator_for3d(b,B,e,E,c,C, 1, {
	  to_v(c,e,b) += from_v(c,e,b);
	});      
    }
  }
}

template<typename FloatType>
void MultiHeadAttentionComponent<FloatType>::update(int off, const Vector<FloatType> &new_params){
  int p=off;
  multW_O.update(p,new_params);
  p += multW_O.nparams();
  for(int h=0;h<heads.size();h++){
    heads[h]->update(p,new_params);
    p += heads[h]->nparams();
  }
}

template<typename FloatType>
void MultiHeadAttentionComponent<FloatType>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  int p=off;
  multW_O.step(p,derivs,eps);
  p += multW_O.nparams();
  for(int h=0;h<heads.size();h++){
    heads[h]->step(p,derivs,eps);
    p += heads[h]->nparams();
  }
}
template<typename FloatType>
void MultiHeadAttentionComponent<FloatType>::getParams(Vector<FloatType> &into, int off){
  int p=off;
  multW_O.getParams(into,p);
  p += multW_O.nparams();
  for(int h=0;h<heads.size();h++){
    heads[h]->getParams(into,p);
    p += heads[h]->nparams();
  }
}

template<typename FloatType>
void MultiHeadAttentionComponent<FloatType>::resizeInputBuffer(size_t to){
  multW_O.resizeInputBuffer(to);
  for(int h=0;h<heads.size();h++)
    heads[h]->resizeInputBuffer(to);
}
