#pragma once
#include <Graph.hpp>

/**
 * @brief A component to extract the inputs to the node update (the aggregated edhes, the node and the global attributes as a 3-tensor)
 */
template<typename Config>
class ExtractNodeUpdateInputComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  bool setup;
  GraphInitialize ginit;
  int tens_size[3];
  Matrix<elemCopyTemplate> copy_template;
public:
  
  ExtractNodeUpdateInputComponent(): setup(false){}

  /**
   * @brief Given a graph containing aggregated edges (one edge per node), return a 3-tensor indexed as [node_idx][stacked_attributes][batch_idx]
   * The aggregated edges of the input should have their node labels set as recv_node == node,  send_node == -1
   */
  Tensor<FloatType,3> value(const Graph<FloatType> &in);
  
  void deriv(Tensor<FloatType,3> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return 0; }

  /**
   * @brief Return the output 3-tensor size
   */
  static std::array<int,3> outputTensorSize(const GraphInitialize &ginit);
};

/**
 * @brief A component to merge the outputs of the node update (the new nodes as a 3-tensor) into the graph
 */
template<typename Config>
struct InsertNodeUpdateOutputComponent{
public:
  EXTRACT_CONFIG_TYPES;
private:
  bool setup;
  GraphInitialize ginit;
  int tens_size[3];

public:
  InsertNodeUpdateOutputComponent(): setup(false){}
  
  /**
   * @brief Replace the nodes in the input graph with those in the provided 3-tensor indexed by [node_idx][stacked_edge_attribs][batch_idx]
   */
  Graph<FloatType> value(const Graph<FloatType> &in, const Tensor<FloatType,3> &node_attr_update);
  
  void deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,3> &dCost_by_dIn_tens) const;

  inline int nparams() const{ return 0; }

  size_t FLOPS(int value_or_deriv) const{ return 0; }

  /**
   * @brief Return the shape of the input tensor
   */
  static std::array<int,3> inputTensorSize(const GraphInitialize &ginit);
};

#include "implementation/NodeUpdate.tcc"
