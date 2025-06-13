#pragma once
#include "LayerCommon.hpp"

//A layer that joins the outputs of two chains into an std::pair. Both chains must ultimately consume the same input
template<typename _FloatType, typename _InputType, typename Store1, typename Store2>
class PairJoinLayer{
  typedef typename Store1::type StoredType1;
  typedef typename Store2::type StoredType2;
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LAYERTYPEOUTPUTTYPE(StoredType1) LayerInputType1;
  typedef LAYERTYPEOUTPUTTYPE(StoredType2) LayerInputType2;
  typedef std::pair<LayerInputType1,LayerInputType2> LayerOutputType;
  typedef LeafTag tag;
private:
  Store1 leaf1;
  Store2 leaf2;
public:
  
  PairJoinLayer(Store1 &&leaf1, Store2 &&leaf2): leaf1(std::move(leaf1)), leaf2(std::move(leaf2)){}
  PairJoinLayer(const PairJoinLayer &r) = delete;
  PairJoinLayer(PairJoinLayer &&r) = default;
  
  inline LayerOutputType value(const InputType &x){
    return LayerOutputType(leaf1.v.value(x),leaf2.v.value(x));
  }
  inline int deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    LayerOutputType above_deriv(std::move(_above_deriv));
    off = leaf1.v.deriv(cost_deriv,off,std::move(above_deriv.first), input_above_deriv_return);
    return leaf2.v.deriv(cost_deriv,off,std::move(above_deriv.second), input_above_deriv_return);
  }
    
  inline int update(int off, const Vector<FloatType> &new_params){
    off = leaf1.v.update(off,new_params);
    return leaf2.v.update(off,new_params);
  }
  
  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    off = leaf1.v.step(off,derivs,eps);
    return leaf2.v.step(off,derivs,eps);
  }

  inline int nparams() const{ return leaf1.v.nparams() + leaf2.v.nparams(); }

  inline size_t FLOPS(int value_or_deriv) const{ return leaf1.v.FLOPS(value_or_deriv) + leaf2.v.FLOPS(value_or_deriv); }
  
  inline int getParams(Vector<FloatType> &into, int off){
    off = leaf1.v.getParams(into,off);
    return leaf2.v.getParams(into,off);
  }

  inline void resizeInputBuffer(size_t to){
    leaf1.v.resizeInputBuffer(to);
    leaf2.v.resizeInputBuffer(to);
  }

};

template<typename U, typename V, typename std::enable_if<ISLEAF(U) && ISLEAF(V) && std::is_same<INPUTTYPE(U),INPUTTYPE(V)>::value , int>::type = 0>
auto pair_join_layer(U &&u, V &&v){
  return PairJoinLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u),DDST(v)>(std::forward<U>(u),std::forward<V>(v));
}
