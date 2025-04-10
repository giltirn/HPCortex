#pragma once
#include<memory>
#include<Layers.hpp>

//This functionality allows dynamic rather than compile time composition of layers
template<typename FloatType, typename InputType, typename LayerOutputType>
class LayerWrapperInternalBase{
public:
  virtual LayerOutputType value(const InputType &x) = 0;
  virtual void deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&above_deriv, InputType* input_above_deriv_return = nullptr) const = 0;
  virtual int nparams() const = 0;
  virtual void resizeInputBuffer(size_t to) = 0;
  virtual void getParams(Vector<FloatType> &into, int off) = 0;
  virtual void step(int off, const Vector<FloatType> &derivs, FloatType eps) = 0;
  virtual ~LayerWrapperInternalBase(){}
};
template<typename Store, typename std::enable_if<ISSTORAGE(Store), int>::type = 0 >
class LayerWrapperInternal: public LayerWrapperInternalBase<typename Store::type::FloatType,
							    typename Store::type::InputType,
							    LAYEROUTPUTTYPE(typename Store::type)>{
public:
  typedef typename Store::type::FloatType FloatType;
  typedef typename Store::type::InputType InputType;
  typedef LAYEROUTPUTTYPE(typename Store::type) LayerOutputType;

  Store layer;
public:
  LayerWrapperInternal(Store &&layer): layer(std::move(layer)){}
  
  LayerOutputType value(const InputType &x) override{
    return layer.v.value(x);
  }
  void deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&above_deriv, InputType* input_above_deriv_return = nullptr) const override{
    layer.v.deriv(cost_deriv,off,std::move(above_deriv), input_above_deriv_return);
  }
  int nparams() const override{ return layer.v.nparams(); }

  void getParams(Vector<FloatType> &into, int off) override{ layer.v.getParams(into,off); }

  void step(int off, const Vector<FloatType> &derivs, FloatType eps) override{ layer.v.step(off,derivs,eps); }
  
  void resizeInputBuffer(size_t to) override{ layer.v.resizeInputBuffer(to); }
};
template<typename _FloatType, typename _InputType, typename _LayerOutputType>
class LayerWrapper{
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef _LayerOutputType LayerOutputType;
private:
  std::unique_ptr<LayerWrapperInternalBase<FloatType,InputType,LayerOutputType> > layer;
public:
  typedef LeafTag tag;

  LayerWrapper(LayerWrapper &&r) = default;
  LayerWrapper & operator=(LayerWrapper &&r) = default;
  
  template<typename Store, typename std::enable_if<ISSTORAGE(Store), int>::type = 0 >
  LayerWrapper(Store &&layer): layer( new LayerWrapperInternal<Store>(std::move(layer)) ){}

  inline LayerOutputType value(const InputType &x){
    return layer->value(x);
  }
  inline void deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&above_deriv, InputType* input_above_deriv_return = nullptr) const{
    layer->deriv(cost_deriv,off, std::move(above_deriv), input_above_deriv_return);
  }
  inline int nparams() const{ return layer->nparams(); }

  inline void getParams(Vector<FloatType> &into, int off){ return layer->getParams(into,off); }

  inline void step(int off, const Vector<FloatType> &derivs, FloatType eps){ return layer->step(off,derivs,eps); }
  
  inline void resizeInputBuffer(size_t to){ layer->resizeInputBuffer(to); }
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
LayerWrapper<FLOATTYPE(U),INPUTTYPE(U),LAYEROUTPUTTYPE(U)> enwrap(U &&u){
  return LayerWrapper<FLOATTYPE(U),INPUTTYPE(U),LAYEROUTPUTTYPE(U)>(DDST(u)(std::forward<U>(u)));
}
