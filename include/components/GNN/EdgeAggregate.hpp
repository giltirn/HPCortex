#pragma once
#include <Graph.hpp>

/**
 * @brief Perform edge aggregation for a node by summing attributes over all connected edges
 */
template<typename Config>
class EdgeAggregateSumComponent{
  int nnode;
  int nedge_attr;
  ManagedTypeArray<Vector<int> > receive_map;
  GraphInitialize ginit;
  GraphInitialize ginit_out;
  bool setup;
  FLOPScounter value_flops;
public:
  EXTRACT_CONFIG_TYPES;
  EdgeAggregateSumComponent(): setup(false){}

  /**
   * @brief Output a new graph where the edge vector has been replaced by aggregated edges, one per node. 
   *        The send_node index in these will be set to -1 to indicate that they are aggregated edges
   */
  Graph<FloatType> value(const Graph<FloatType> &graph);
  
  void deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return value_or_deriv == 0 ? value_flops.value() : 0; }
};

/**
 * @brief Perform edge aggregation for a global attribute update by summing attributes over all edges
 */
template<typename Config>
class EdgeAggregateGlobalSumComponent{
  int nedge_attr;
  GraphInitialize ginit;
  GraphInitialize ginit_out;
  bool setup;
  FLOPScounter value_flops;
public:
  EXTRACT_CONFIG_TYPES;
  EdgeAggregateGlobalSumComponent(): setup(false){}

    /**
   * @brief Output a new graph where the edge vector has been replaced a single aggregated edge
   *        The send_node and recv_node index in these will be set to -1 to indicate that they are aggregated edges
   */
  Graph<FloatType> value(const Graph<FloatType> &graph);
  
  void deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return value_or_deriv == 0 ? value_flops.value() : 0; }
};

#include "implementation/EdgeAggregate.tcc"
