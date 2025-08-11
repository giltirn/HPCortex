#pragma once
#include <Graph.hpp>

/**
 * @brief A component to extract the inputs to the global update (the aggregated edges and nodes, and the global attributes as a 3-tensor)
 */
template<typename Config>
class ExtractGlobalUpdateInputComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  bool setup;
  GraphInitialize ginit;
  int tens_size[2];

public:
  ExtractGlobalUpdateInputComponent(): setup(false){}

  /**
   * @brief Return a 2-tensor indexed as [attribute_idx][batch_idx]
   */
  Tensor<FloatType,2> value(const Graph<FloatType> &gaggr);

  void deriv(Tensor<FloatType,2> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return 0; }

  /**
   * @brief Return the output 2-tensor size
   */
  static std::array<int,2> outputTensorSize(const GraphInitialize &ginit);
};

/**
 * @brief A component to merge the outputs of the global update (the new global attributes) into the graph
 */
template<typename Config>
class InsertGlobalUpdateOutputComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  bool setup;
  GraphInitialize ginit;
  int global_attr_size_total;
public:
  
  InsertGlobalUpdateOutputComponent(): setup(false){}
  
  Graph<FloatType> value(const Graph<FloatType> &in, const Tensor<FloatType,2> &global_attr_update);
  
  void deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,2> &dCost_by_dIn_tens) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return 0; }

  static std::array<int,2> inputTensorSize(const GraphInitialize &ginit);
};


#include "implementation/GlobalUpdate.tcc"
