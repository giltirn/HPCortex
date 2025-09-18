#pragma once
#include <Graph.hpp>

/**
 * @brief A component to extract the inputs to the edge update (the egdes, their attached nodes and the global attributes as a 3-tensor
 */
template<typename Config>
class ExtractEdgeUpdateInputComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  bool setup;
  GraphInitialize ginit;
  int nedge;
  int tens_size[3];
  Matrix<elemCopyTemplate> copy_template;
public:
  
  ExtractEdgeUpdateInputComponent(): setup(false){}

  /** 
   * @brief Return a 3-tensor indexed as [edge_idx][stacked_attributes][batch_idx]
   */
  Tensor<FloatType,3> value(const Graph<FloatType> &in);
  
  void deriv(Tensor<FloatType,3> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return 0; }

  /**
   * @brief Return the shape of the output tensor
   */
  static std::array<int,3> outputTensorSize(const GraphInitialize &ginit);
};

/**
 * @brief A component to merge the outputs of the edge update (the new edges as a 3-tensor) into the graph
 */
template<typename Config>
class InsertEdgeUpdateOutputComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  bool setup;
  GraphInitialize ginit;
  int tens_size[3];
public:
  InsertEdgeUpdateOutputComponent(): setup(false){}

  /**
   * @brief Replace the edges in the input graph with those in the provided 3-tensor indexed by [edge_idx][stacked_edge_attribs][batch_idx]
   */
  Graph<FloatType> value(const Graph<FloatType> &in, const Tensor<FloatType,3> &edge_attr_update);

  void deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,3> &dCost_by_dIn_tens) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return 0; }

  /**
   * @brief Return the size of the input 3-tensor
   */
  static std::array<int,3> inputTensorSize(const GraphInitialize &ginit);
};

#include "implementation/EdgeUpdate.tcc"
