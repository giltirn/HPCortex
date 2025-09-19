#pragma once
#include <Graph.hpp>

/**
 * @brief Perform node aggregation for a global attribute update by summing attributes over all nodes
 */
template<typename Config>
class NodeAggregateGlobalSumComponent{
  GraphInitialize ginit;
  GraphInitialize ginit_out;
  bool setup;
  FLOPScounter value_flops;
  int nnode_attr;
public:
  EXTRACT_CONFIG_TYPES;
  NodeAggregateGlobalSumComponent(): setup(false){}

  /**
   * @brief Output a new graph where the node vector has been replaced a single aggregated node
   */
  template<typename InGraphType, enable_if_fwd_ref<InGraphType,Graph<FloatType> > =0 >
  Graph<FloatType> value(InGraphType &&graph);
  
  void deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return value_or_deriv == 0 ? value_flops.value() : 0; }
};

#include "implementation/NodeAggregate.tcc"
