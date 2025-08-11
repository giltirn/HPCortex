#pragma once
#include "LayerCommon.hpp"
#include "ComponentLayerWrapper.hpp"
#include <components/GNN.hpp>

/**
 * @brief Wrap GCN components in layers
 */
DECLARE_BASIC_COMPONENT_LAYER_WRAPPER(ExtractEdgeUpdateInputLayer, extract_edge_update_input_layer, ExtractEdgeUpdateInputComponent)
DECLARE_BASIC_MERGE_COMPONENT_LAYER_WRAPPER(InsertEdgeUpdateOutputLayer, insert_edge_update_output_layer, InsertEdgeUpdateOutputComponent);
DECLARE_BASIC_COMPONENT_LAYER_WRAPPER(EdgeAggregateSumLayer, edge_aggregate_sum_layer, EdgeAggregateSumComponent)
DECLARE_BASIC_COMPONENT_LAYER_WRAPPER(EdgeAggregateGlobalSumLayer, edge_aggregate_global_sum_layer, EdgeAggregateGlobalSumComponent)
DECLARE_BASIC_COMPONENT_LAYER_WRAPPER(ExtractNodeUpdateInputLayer, extract_node_update_input_layer, ExtractNodeUpdateInputComponent)
DECLARE_BASIC_MERGE_COMPONENT_LAYER_WRAPPER(InsertNodeUpdateOutputLayer, insert_node_update_output_layer, InsertNodeUpdateOutputComponent);
DECLARE_BASIC_COMPONENT_LAYER_WRAPPER(NodeAggregateGlobalSumLayer, node_aggregate_global_sum_layer, NodeAggregateGlobalSumComponent)
DECLARE_BASIC_COMPONENT_LAYER_WRAPPER(ExtractGlobalUpdateInputLayer, extract_global_update_input_layer, ExtractGlobalUpdateInputComponent)
DECLARE_BASIC_MERGE_COMPONENT_LAYER_WRAPPER(InsertGlobalUpdateOutputLayer, insert_global_update_output_layer, InsertGlobalUpdateOutputComponent);

/**
 * @brief Namespace for GCN sub-blocks
 */
namespace GCN {
  /**
   * @brief Construct the edge update sub-block
   */
  template<typename Below, typename EdgeUpdateBlockInitializer>
  auto edgeUpdateBlock(const GraphInitialize &ginit, const EdgeUpdateBlockInitializer &einit, Below &&below);
  /**
   * @brief Construct the node update sub-block
   */
  template<typename Below, typename NodeUpdateBlockInitializer>
  auto nodeUpdateBlock(const GraphInitialize &ginit, const NodeUpdateBlockInitializer &ninit, Below &&below);

  /**
   * @brief Construct the global update sub-block
   */

  template<typename Below, typename GlobalUpdateBlockInitializer>
  auto globalUpdateBlock(const GraphInitialize &ginit, const GlobalUpdateBlockInitializer &glinit, Below &&below);

};

/**
 * @brief Create a graph convolutional network (GCN) block. Initializer lambdas should be provided to build the update operations
          Initializer objects should have signature   [](int fan_out, int fan_in, auto &&in)
	  where fan_in/fan_out are the dimensions of the tensor dimension that the layer should operate on. 
	  The blocks create by EdgeUpdateBlockInitializer and NodeUpdateBlockInitializer should both accept and return 3-tensors and act on the second dimension (1)
	  The block created by GlobalUpdateBlockInitializer should accept and return 2-tensors and act on the first dimension (0)
*/
template<typename Below, typename EdgeUpdateBlockInitializer, typename NodeUpdateBlockInitializer, typename GlobalUpdateBlockInitializer >
auto GCNblock(const GraphInitialize &ginit, const EdgeUpdateBlockInitializer &einit, const NodeUpdateBlockInitializer &ninit, const GlobalUpdateBlockInitializer &glinit, Below &&below){
  auto eup = GCN::edgeUpdateBlock(ginit, einit, std::forward<Below>(below));
  auto nup = GCN::nodeUpdateBlock(ginit, ninit, std::move(eup));
  return GCN::globalUpdateBlock(ginit, glinit, std::move(nup));  
}

#include "implementation/GCNblock.tcc"
