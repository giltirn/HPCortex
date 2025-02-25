#pragma once
#include<memory>
#include<Layers.hpp>

//This functionality allows dynamic rather than compile time composition of layers
class LayerWrapperInternalBase{
public:
  virtual Matrix value(const Matrix &x) = 0;
  virtual void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv, Matrix* input_above_deriv_copyback = nullptr) const = 0;
  virtual int nparams() const = 0;
  virtual void resizeInputBuffer(size_t to) = 0;
  virtual void getParams(Vector &into, int off) = 0;
  virtual void step(int off, const Vector &derivs, double eps) = 0;
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
  int nparams() const override{ return layer.v.nparams(); }

  void getParams(Vector &into, int off) override{ layer.v.getParams(into,off); }

  void step(int off, const Vector &derivs, double eps) override{ layer.v.step(off,derivs,eps); }
  
  void resizeInputBuffer(size_t to) override{ layer.v.resizeInputBuffer(to); }
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

  inline void getParams(Vector &into, int off){ return layer->getParams(into,off); }

  inline void step(int off, const Vector &derivs, double eps){ return layer->step(off,derivs,eps); }
  
  inline void resizeInputBuffer(size_t to){ layer->resizeInputBuffer(to); }
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
LayerWrapper enwrap(U &&u){
  return LayerWrapper(DDST(u)(std::forward<U>(u)));
}
