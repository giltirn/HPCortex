#pragma once
#include "LayerCommon.hpp"

template<typename _FloatType, typename _InputType, typename Store>
class ReplicateLayerLeader{
  public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef typename Store::type StoredType;
  typedef LAYERTYPEOUTPUTTYPE(StoredType) LayerInputOutputType;
  LayerInputOutputType in_buf;
  std::vector<LayerInputOutputType> above_deriv;
  
  Store leaf;
  int N;
  int val_count;
  int deriv_count;
  int update_count;
  int step_count;
  int getparams_count;

  ReplicateLayerLeader(Store &&leaf, int N): leaf(std::move(leaf)), N(N), val_count(0), deriv_count(0), update_count(0), step_count(0), getparams_count(0), above_deriv(N){}

  inline void cinc(int &i){ i = (i+1) % N; }
  
  LayerInputOutputType value(const InputType &x){
    if(val_count == 0)
      in_buf = leaf.v.value(x); //x is assumed to be the same for all calls from children (not checked)
    cinc(val_count);
    return in_buf;    
  }
  
  //we want to allow for children to call deriv,update,step etc in any order, although the order must remain consistent to preserve the indexing of the parameters in the parameter vector
  
  int deriv(Vector<FloatType> &cost_deriv, int off, LayerInputOutputType &&_above_deriv, InputType* input_above_deriv_return){
    above_deriv[deriv_count] = std::move(_above_deriv);
    int ret = off;
    if(deriv_count == N-1){
      //Out^(i) = x
      //dCost / dx_j = \sum_i  dCost/Out^(i)_j  -> above_deriv for layer below
      //             = \sum_i  dCost/Out^(i)_j 
      for(int i=1;i<N;i++){
	above_deriv[0] += above_deriv[i];
      }
      ret = leaf.v.deriv(cost_deriv, off, std::move(above_deriv[0]), input_above_deriv_return ); //last instance will have the correct offset assuming 
    }    
    cinc(deriv_count);
    return ret;
  }
  int update(int off, const Vector<FloatType> &new_params){
    int ret = off;
    if(update_count == N-1)
      ret = leaf.v.update(off, new_params);
    cinc(update_count);
    return ret;
  }
  int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    int ret = off;
    if(step_count == N-1)
      ret = leaf.v.step(off, derivs, eps);
    cinc(step_count);
    return ret;
  }
  int getParams(Vector<FloatType> &into, int off){
    int ret = off;
    if(getparams_count == N-1)
      ret = leaf.v.getParams(into, off);
    cinc(getparams_count);
    return ret;
  }
  
};



template<typename _FloatType, typename _InputType, typename Store>
class ReplicateLayer{
  typedef typename Store::type StoredType;
public:
  typedef _FloatType FloatType;
  typedef _InputType InputType;
  typedef LAYERTYPEOUTPUTTYPE(StoredType) LayerInputOutputType;
private:
  int instance;
  int N;
  ReplicateLayerLeader<FloatType,InputType,Store> *leader;
public:
  typedef LeafTag tag;
  
  ReplicateLayer(ReplicateLayerLeader<FloatType,InputType,Store> *leader, int instance, int N): leader(leader), instance(instance), N(N){}
  ReplicateLayer(const ReplicateLayer &r) = delete;
  ReplicateLayer(ReplicateLayer &&r): instance(r.instance), N(r.N), leader(r.leader){
    r.leader = nullptr;
  }

  ~ReplicateLayer(){
    if(!instance && leader != nullptr) delete leader; //first in group owns the pointer
  }
  
  inline LayerInputOutputType value(const InputType &x){
    return leader->value(x);
  }
  inline int deriv(Vector<FloatType> &cost_deriv, int off, LayerInputOutputType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const{
    return leader->deriv(cost_deriv,off,std::move(_above_deriv), input_above_deriv_return);
  }
    
  inline int update(int off, const Vector<FloatType> &new_params){
    return leader->update(off,new_params);
  }
  
  inline int step(int off, const Vector<FloatType> &derivs, FloatType eps){
    return leader->step(off,derivs, eps);
  }

  inline int nparams() const{ return instance == N-1 ? leader->leaf.v.nparams() : 0; }

  inline size_t FLOPS(int value_or_deriv) const{ return instance == N-1 ? leader->leaf.v.FLOPS(value_or_deriv) : 0; }
  
  inline int getParams(Vector<FloatType> &into, int off) const{
    return leader->getParams(into,off);
  }

  inline void resizeInputBuffer(size_t to){
    if(!instance) leader->leaf.v.resizeInputBuffer(to);
  }

};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto replicate_layer(int N,
		     U &&u){
  typedef ReplicateLayer<FLOATTYPE(U),INPUTTYPE(U),DDST(u)> Branch;
  typedef ReplicateLayerLeader<FLOATTYPE(U),INPUTTYPE(U),DDST(u)> Leader;

  Leader* leader = new Leader(std::forward<U>(u), N);
  std::vector<std::unique_ptr<Branch> > out(N);
  for(int i=0;i<N;i++)
    out[i].reset( new Branch(leader, i, N) );
  return out;
}
