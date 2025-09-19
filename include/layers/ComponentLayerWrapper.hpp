#pragma once
#include "LayerCommon.hpp"

/**
 * @brief A class to wrap "basic" components (those that are default-constructible and without trainable parameters) as a layer
 *        These components must not require internal buffer storage, and their "value" function must not accept EnableDeriv argument
 */
template<typename Config, typename _InputType, typename ComponentType, typename BelowStore>
class BasicComponentLayerWrapper{
public:
  EXTRACT_CONFIG_TYPES;
  typedef _InputType InputType;
  typedef typename BelowStore::type BelowLayerType;
  typedef LAYEROUTPUTTYPE(BelowLayerType) LayerInputType;
  typedef decltype( std::declval<ComponentType>().value( std::declval<LayerInputType>() ) ) LayerOutputType;
private:
  ComponentType cpt;
  BelowStore leaf;  
public:
  typedef LeafTag tag;
  
  BasicComponentLayerWrapper(BelowStore &&leaf): leaf(std::move(leaf)){
    assert(cpt.nparams() == 0); //basic!
  }
  BasicComponentLayerWrapper(BasicComponentLayerWrapper &&r) = default;
  BasicComponentLayerWrapper(const BasicComponentLayerWrapper &r) = delete;

  LayerOutputType value(const InputType &x, EnableDeriv enable_deriv = DerivNo){
    return cpt.value(leaf.v.value(x, enable_deriv));    
  }

  //input_above_deriv_return is the derivative of the cost with respect to the inputs
  int deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&above_deriv, InputType* input_above_deriv_return = nullptr) const{
    LayerInputType in_deriv;
    cpt.deriv(std::move(above_deriv), in_deriv);
    return leaf.v.deriv(cost_deriv, off, std::move(in_deriv), input_above_deriv_return);
  }
  
  inline int update(int off, const Vector<FloatType> &new_params){ return leaf.v.update(off, new_params); }

  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){ return leaf.v.step(off, derivs, eps); }
  
  inline int nparams() const{ return leaf.v.nparams(); }

  inline size_t FLOPS(int value_or_deriv) const{ return cpt.FLOPS(value_or_deriv) + leaf.v.FLOPS(value_or_deriv); }
  
  inline int getParams(Vector<FloatType> &into, int off) const{ return leaf.v.getParams(into,off); }

  inline void resizeInputBuffer(size_t to){ leaf.v.resizeInputBuffer(to); }

  inline ComponentType &getComponent() const{ return cpt; }
};

#define DECLARE_BASIC_COMPONENT_LAYER_WRAPPER(LAYER_NAME, LAYER_FUNC_NAME, COMPONENT_TYPE) \
\
template<typename Config, typename InputType, typename BelowStore> \
using LAYER_NAME = BasicComponentLayerWrapper<Config,InputType, COMPONENT_TYPE<Config>,BelowStore>; \
\
\
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0> \
auto LAYER_FUNC_NAME(U &&u){ \
  return LAYER_NAME <CONFIGTYPE(U),INPUTTYPE(U),DDST(u)>(std::forward<U>(u)); \
}





/**
 * @brief A class to wrap "basic" components (those that are default-constructible and without trainable parameters) that merge two inputs as a layer
 *        These components must not require internal buffer storage, and their "value" function must accept two arguments corresponding to the outputs of the leaf chains
 */
template<typename Config, typename _InputType, typename ComponentType, typename BelowStore1, typename BelowStore2>
class BasicMergeComponentLayerWrapper{
public:
  EXTRACT_CONFIG_TYPES;
  typedef _InputType InputType;
  
  typedef typename BelowStore1::type BelowLayerType1;  
  typedef LAYEROUTPUTTYPE(BelowLayerType1) LayerInputType1;

  typedef typename BelowStore2::type BelowLayerType2;  
  typedef LAYEROUTPUTTYPE(BelowLayerType2) LayerInputType2;
  
  typedef decltype( std::declval<ComponentType>().value( std::declval<LayerInputType1>(), std::declval<LayerInputType2>() ) ) LayerOutputType;
private:
  ComponentType cpt;
  BelowStore1 leaf1;
  BelowStore2 leaf2;
public:
  typedef LeafTag tag;
  
  BasicMergeComponentLayerWrapper(BelowStore1 &&leaf1, BelowStore2 &&leaf2): leaf1(std::move(leaf1)), leaf2(std::move(leaf2)){
    assert(cpt.nparams() == 0); //basic!
  }
  BasicMergeComponentLayerWrapper(BasicMergeComponentLayerWrapper &&r) = default;
  BasicMergeComponentLayerWrapper(const BasicMergeComponentLayerWrapper &r) = delete;

  LayerOutputType value(const InputType &x, EnableDeriv enable_deriv = DerivNo){
    auto v1 = leaf1.v.value(x,enable_deriv);
    auto v2 = leaf2.v.value(x,enable_deriv);    
    return cpt.value(std::move(v1),std::move(v2));    
  }

  //input_above_deriv_return is the derivative of the cost with respect to the inputs
  int deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&above_deriv, InputType* input_above_deriv_return = nullptr) const{
    LayerInputType1 in_deriv1;
    LayerInputType2 in_deriv2;
    cpt.deriv(std::move(above_deriv), in_deriv1, in_deriv2);
    off = leaf1.v.deriv(cost_deriv, off, std::move(in_deriv1), input_above_deriv_return);
    return leaf2.v.deriv(cost_deriv, off, std::move(in_deriv2), input_above_deriv_return);
  }
  
  inline int update(int off, const Vector<FloatType> &new_params){
    off = leaf1.v.update(off, new_params);
    return leaf2.v.update(off, new_params);
  }

  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    off = leaf1.v.step(off, derivs, eps);
    return leaf2.v.step(off, derivs, eps);
  }
  
  inline int nparams() const{ return leaf1.v.nparams() + leaf2.v.nparams(); }

  inline size_t FLOPS(int value_or_deriv) const{ return cpt.FLOPS(value_or_deriv) + leaf1.v.FLOPS(value_or_deriv) + leaf2.v.FLOPS(value_or_deriv); }
  
  inline int getParams(Vector<FloatType> &into, int off) const{
    off = leaf1.v.getParams(into,off);
    return leaf2.v.getParams(into,off);
  }
  inline void resizeInputBuffer(size_t to){
    leaf1.v.resizeInputBuffer(to);
    leaf2.v.resizeInputBuffer(to);
  }

  inline ComponentType &getComponent() const{ return cpt; }
};

#define DECLARE_BASIC_MERGE_COMPONENT_LAYER_WRAPPER(LAYER_NAME, LAYER_FUNC_NAME, COMPONENT_TYPE) \
\
  template<typename Config, typename InputType, typename BelowStore1, typename BelowStore2>	\
 using LAYER_NAME = BasicMergeComponentLayerWrapper<Config,InputType, COMPONENT_TYPE<Config>,BelowStore1,BelowStore2>; \
 \
 \
  template<typename U, typename V, typename std::enable_if<ISLEAF(U) && ISLEAF(V), int>::type = 0> \
  auto LAYER_FUNC_NAME(U &&u, V &&v){						\
    return LAYER_NAME <CONFIGTYPE(U),INPUTTYPE(U),DDST(u),DDST(v)>(std::forward<U>(u),std::forward<V>(v)); \
 }
