#pragma once
#include<memory>
#include<Layers.hpp>

//This functionality allows dynamic rather than compile time composition of layers
class LayerWrapperInternalBase{
public:
  virtual Matrix value(const Matrix &x) = 0;
  virtual void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const = 0;
  virtual int nparams() const = 0;
  virtual ~LayerWrapperInternalBase(){}
};
template<typename Store, typename std::enable_if<ISSTORAGE(Store), int>::type = 0 >
class LayerWrapperInternal: public LayerWrapperInternalBase{
  Store layer;
public:
  LayerWrapperInternal(Store &&layer): layer(std::move(layer)){}
  
  Matrix value(const Matrix &x) override{
    return layer.v.value(x);
  }
  void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const override{
    layer.v.deriv(cost_deriv,off,above_deriv, input_above_deriv_copyback);
  }
  int nparams() const{ return layer.v.nparams(); }
};
class LayerWrapper{
  std::unique_ptr<LayerWrapperInternalBase> layer;
public:
  typedef LeafTag tag;

  LayerWrapper(LayerWrapper &&r) = default;
  LayerWrapper & operator=(LayerWrapper &&r) = default;
  
  template<typename Store, typename std::enable_if<ISSTORAGE(Store), int>::type = 0 >
  LayerWrapper(Store &&layer): layer( new LayerWrapperInternal<Store>(std::move(layer)) ){}

  inline Matrix value(const Matrix &x){
    return layer->value(x);
  }
  inline void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const{
    layer->deriv(cost_deriv,off,above_deriv, input_above_deriv_copyback);
  }
  inline int nparams() const{ return layer->nparams(); }
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
LayerWrapper enwrap(U &&u){
  return LayerWrapper(DDST(u)(std::forward<U>(u)));
}
