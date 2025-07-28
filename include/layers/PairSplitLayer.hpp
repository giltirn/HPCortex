#pragma once
#include "LayerCommon.hpp"

template<typename Config, typename _InputType, typename Store>
class PairSplitLayerLeader{
public:
  EXTRACT_CONFIG_TYPES;
  typedef _InputType InputType;
  typedef typename Store::type StoredType;
  typedef LAYERTYPEOUTPUTTYPE(StoredType) LayerInputType;
  typedef typename LayerInputType::first_type LayerOutputType1;
  typedef typename LayerInputType::second_type LayerOutputType2;
    
  LayerInputType in_buf;
  LayerOutputType1 above_deriv1;
  LayerOutputType2 above_deriv2;
  
  Store leaf;
  int val_count;
  int deriv_count;
  int update_count;
  int step_count;
  int getparams_count;

  PairSplitLayerLeader(Store &&leaf): leaf(std::move(leaf)), val_count(0), deriv_count(0), update_count(0), step_count(0), getparams_count(0){}

  inline void cinc(int &i){ i = (i+1) % 2; }
  
  LayerOutputType1 first(const InputType &x, EnableDeriv enable_deriv){
    if(val_count == 0)
      in_buf = leaf.v.value(x,enable_deriv); //x is assumed to be the same for all calls from children (not checked)
    cinc(val_count);
    return in_buf.first;    
  }
  LayerOutputType2 second(const InputType &x, EnableDeriv enable_deriv){
    if(val_count == 0)
      in_buf = leaf.v.value(x,enable_deriv);
    cinc(val_count);
    return in_buf.second;    
  }
    
  //we want to allow for children to call deriv,update,step etc in any order, although the order must remain consistent to preserve the indexing of the parameters in the parameter vector
  int deriv_complete(Vector<FloatType> &cost_deriv, int off, InputType* input_above_deriv_return){
    LayerInputType der(std::move(above_deriv1), std::move(above_deriv2));
    return leaf.v.deriv(cost_deriv, off, std::move(der), input_above_deriv_return ); 
  }
    
  int deriv_first(Vector<FloatType> &cost_deriv, int off, LayerOutputType1 &&_above_deriv, InputType* input_above_deriv_return){
    above_deriv1 = std::move(_above_deriv);
    int ret = off;
    if(deriv_count == 1)
      ret = deriv_complete(cost_deriv, off, input_above_deriv_return);
    cinc(deriv_count);
    return ret;
  }
  int deriv_second(Vector<FloatType> &cost_deriv, int off, LayerOutputType2 &&_above_deriv, InputType* input_above_deriv_return){
    above_deriv2 = std::move(_above_deriv);
    int ret = off;
    if(deriv_count == 1)
      ret = deriv_complete(cost_deriv, off, input_above_deriv_return);
    cinc(deriv_count);
    return ret;
  }
  
  int update(int off, const Vector<FloatType> &new_params){
    int ret = off;
    if(update_count == 1)
      ret = leaf.v.update(off, new_params);
    cinc(update_count);
    return ret;
  }
  int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    int ret = off;
    if(step_count == 1)
      ret = leaf.v.step(off, derivs, eps);
    cinc(step_count);
    return ret;
  }
  int getParams(Vector<FloatType> &into, int off){
    int ret = off;
    if(getparams_count == 1)
      ret = leaf.v.getParams(into, off);
    cinc(getparams_count);
    return ret;
  }
  
};



template<typename Config, typename _InputType, typename Store>
class PairSplitLayer1{
  typedef typename Store::type StoredType;
public:
  EXTRACT_CONFIG_TYPES;
  typedef LAYERTYPEOUTPUTTYPE(StoredType) LayerInputType;
  typedef typename LayerInputType::first_type LayerOutputType;
  typedef _InputType InputType;
private:
  PairSplitLayerLeader<Config,InputType,Store> *leader;
public:
  typedef LeafTag tag;
  
  PairSplitLayer1(PairSplitLayerLeader<Config,InputType,Store> *leader): leader(leader){}
  PairSplitLayer1(const PairSplitLayer1 &r) = delete;
  PairSplitLayer1(PairSplitLayer1 &&r): leader(r.leader){
    r.leader = nullptr;
  }    

  ~PairSplitLayer1(){ //this one owns the pointer
    if(leader != nullptr) delete leader;
  }
  
  inline LayerOutputType value(const InputType &x, EnableDeriv enable_deriv = DerivNo){
    return leader->first(x,enable_deriv);
  }
  inline int deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    return leader->deriv_first(cost_deriv,off,std::move(_above_deriv), input_above_deriv_return);
  }
    
  inline int update(int off, const Vector<FloatType> &new_params){
    return leader->update(off,new_params);
  }
  
  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    return leader->step(off,derivs, eps);
  }

  inline int nparams() const{ return 0; } //other instance handles nparams for leaf

  inline size_t FLOPS(int value_or_deriv) const{ return 0; } //and FLOPS
  
  inline int getParams(Vector<FloatType> &into, int off) const{
    return leader->getParams(into,off);
  }

  inline void resizeInputBuffer(size_t to){
    leader->leaf.v.resizeInputBuffer(to);
  }
};


template<typename Config, typename _InputType, typename Store>
class PairSplitLayer2{
  typedef typename Store::type StoredType;
public:
  EXTRACT_CONFIG_TYPES;
  typedef LAYERTYPEOUTPUTTYPE(StoredType) LayerInputType;
  typedef typename LayerInputType::second_type LayerOutputType;
  typedef _InputType InputType;
private:
  PairSplitLayerLeader<Config,InputType,Store> *leader;
public:
  typedef LeafTag tag;
  
  PairSplitLayer2(PairSplitLayerLeader<Config,InputType,Store> *leader): leader(leader){}
  PairSplitLayer2(const PairSplitLayer2 &r) = delete;
  PairSplitLayer2(PairSplitLayer2 &&r) = default;
  
  inline LayerOutputType value(const InputType &x, EnableDeriv enable_deriv = DerivNo){
    return leader->second(x,enable_deriv);
  }
  inline int deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    return leader->deriv_second(cost_deriv,off,std::move(_above_deriv), input_above_deriv_return);
  }
    
  inline int update(int off, const Vector<FloatType> &new_params){
    return leader->update(off,new_params);
  }
  
  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    return leader->step(off,derivs, eps);
  }

  inline int nparams() const{ return leader->leaf.v.nparams(); }

  inline size_t FLOPS(int value_or_deriv) const{ return leader->leaf.v.FLOPS(value_or_deriv); }
  
  inline int getParams(Vector<FloatType> &into, int off) const{
    return leader->getParams(into,off);
  }

  inline void resizeInputBuffer(size_t to){
  }
};



template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto pair_split_layer(U &&u){
  typedef PairSplitLayer1<CONFIGTYPE(U),INPUTTYPE(U),DDST(u)> Branch1;
  typedef PairSplitLayer2<CONFIGTYPE(U),INPUTTYPE(U),DDST(u)> Branch2;
  typedef PairSplitLayerLeader<CONFIGTYPE(U),INPUTTYPE(U),DDST(u)> Leader;

  Leader* leader = new Leader(std::forward<U>(u));
  std::pair<std::unique_ptr<Branch1>, std::unique_ptr<Branch2> > out;
  out.first.reset(new Branch1(leader));
  out.second.reset(new Branch2(leader));
  return out;
}
