#pragma once
#include "LayerCommon.hpp"

//A layer that joins the outputs of two chains by summation
template<typename _FloatType, typename _InputType, typename Store1, typename Store2>
class SumJoinLayer{
  typedef typename Store1::type StoredType1;
  typedef typename Store2::type StoredType2;

  typedef LAYERTYPEOUTPUTTYPE(StoredType1) LayerInputType1;
  typedef LAYERTYPEOUTPUTTYPE(StoredType2) LayerInputType2;
  static_assert( std::is_same<LayerInputType1, LayerInputType2>::value );  
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  
  typedef LayerInputType1 LayerInputOutputType;
  typedef LeafTag tag;
private:
  Store1 leaf1;
  Store2 leaf2;
public:
  
  SumJoinLayer(Store1 &&leaf1, Store2 &&leaf2): leaf1(std::move(leaf1)), leaf2(std::move(leaf2)){}
  SumJoinLayer(const SumJoinLayer &r) = delete;
  SumJoinLayer(SumJoinLayer &&r) = default;
  
  inline LayerInputOutputType value(const InputType &x){
    return leaf1.v.value(x) + leaf2.v.value(x);
  }
  inline int deriv(Vector<FloatType> &cost_deriv, int off, LayerInputOutputType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    LayerInputOutputType above_deriv(std::move(_above_deriv));
    //Out = X + Y
    //dCost/dX_i = dCost/dOut_i dOut_i/dX_i = dCost/dOut_i   -> above deriv for leaf1
    //dCost/dY_i = dCost/dOut_i dOut_i/dY_i = dCost/dOut_i   -> above deriv for leaf2
    off = leaf1.v.deriv(cost_deriv,off, LayerInputOutputType(above_deriv), input_above_deriv_return);
    return leaf2.v.deriv(cost_deriv,off,std::move(above_deriv), input_above_deriv_return);
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
auto sum_join_layer(U &&u, V &&v){
  return SumJoinLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u),DDST(v)>(std::forward<U>(u),std::forward<V>(v));
}
