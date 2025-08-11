namespace GCN {

  template<typename Below, typename EdgeUpdateBlockInitializer>
  auto edgeUpdateBlock(const GraphInitialize &ginit, const EdgeUpdateBlockInitializer &einit, Below &&below){
    auto splt = replicate_layer(2, std::forward<Below>(below));
    auto eup_in = extract_edge_update_input_layer(std::move(*splt[0]));
    std::array<int,3> eup_in_tens_sz = ExtractEdgeUpdateInputComponent<CONFIGTYPE(Below)>::outputTensorSize(ginit);
    std::array<int,3> eup_out_tens_sz = InsertEdgeUpdateOutputComponent<CONFIGTYPE(Below)>::inputTensorSize(ginit);

    auto eup = einit(eup_out_tens_sz[1], eup_in_tens_sz[1], std::move(eup_in));
    return insert_edge_update_output_layer(std::move(*splt[1]), std::move(eup));
  }


  template<typename Below, typename NodeUpdateBlockInitializer>
  auto nodeUpdateBlock(const GraphInitialize &ginit, const NodeUpdateBlockInitializer &ninit, Below &&below){
    auto splt = replicate_layer(2, std::forward<Below>(below));
    auto agg = edge_aggregate_sum_layer(std::move(*splt[0]));
    auto nup_in = extract_node_update_input_layer(std::move(agg));
    std::array<int,3> nup_in_tens_sz = ExtractNodeUpdateInputComponent<CONFIGTYPE(Below)>::outputTensorSize(ginit);
    std::array<int,3> nup_out_tens_sz = InsertNodeUpdateOutputComponent<CONFIGTYPE(Below)>::inputTensorSize(ginit);

    auto nup = ninit(nup_out_tens_sz[1], nup_in_tens_sz[1], std::move(nup_in));
    return insert_node_update_output_layer(std::move(*splt[1]), std::move(nup));
  }

  template<typename Below, typename GlobalUpdateBlockInitializer>
  auto globalUpdateBlock(const GraphInitialize &ginit, const GlobalUpdateBlockInitializer &glinit, Below &&below){
    auto splt = replicate_layer(2, std::forward<Below>(below));
    auto eagg = edge_aggregate_global_sum_layer(std::move(*splt[0]));
    auto nagg = node_aggregate_global_sum_layer(std::move(eagg));
    auto gup_in = extract_global_update_input_layer(std::move(nagg));
  
    std::array<int,2> gup_in_tens_sz = ExtractGlobalUpdateInputComponent<CONFIGTYPE(Below)>::outputTensorSize(ginit);
    std::array<int,2> gup_out_tens_sz = InsertGlobalUpdateOutputComponent<CONFIGTYPE(Below)>::inputTensorSize(ginit);

    auto gup = glinit(gup_out_tens_sz[0], gup_in_tens_sz[0], std::move(gup_in));
    return insert_global_update_output_layer(std::move(*splt[1]), std::move(gup));
  }

};
