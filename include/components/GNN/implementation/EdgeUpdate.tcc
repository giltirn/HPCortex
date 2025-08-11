template<typename Config>
Tensor<typename Config::FloatType,3> ExtractEdgeUpdateInputComponent<Config>::value(const Graph<FloatType> &in){
  if(!setup){
    ginit = in.getInitializer();
    nedge = ginit.edge_map.size();
    int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
    int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
    int global_attr_size_total =  ginit.totalAttribSize(ginit.global_attr_sizes);
    
    tens_size[0] = ginit.edge_map.size();
    tens_size[1] = 2*node_attr_size_total + edge_attr_size_total + global_attr_size_total;
    tens_size[2] = ginit.batch_size;

    setup = true;
  }

  Tensor<FloatType,3> out(tens_size);
  autoView(out_v,out,DeviceWrite);

  FloatType *to_base = out_v.data();
  for(int edge_idx=0; edge_idx < nedge; edge_idx++){
    const Edge<FloatType> &edge = in.edges[edge_idx];
    const Node<FloatType> &send_node = in.nodes[ edge.send_node ];
    const Node<FloatType> &recv_node = in.nodes[ edge.recv_node ];

    //stack attributes of send node
    to_base = stackAttr(to_base, send_node);
      
    //then receive node      
    to_base = stackAttr(to_base, recv_node);
      
    //then edge
    to_base = stackAttr(to_base, edge);
        
    //then global
    to_base = stackAttr(to_base, in.global);
  }
  return out;
}
  
template<typename Config>
void ExtractEdgeUpdateInputComponent<Config>::deriv(Tensor<FloatType,3> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  Tensor<FloatType,3> in(_dCost_by_dOut);
  Graph<FloatType> &out = dCost_by_dIn;

  assert(setup);
  out = Graph<FloatType>(ginit);
    
  assert(in.size(0) == tens_size[0] && in.size(1) == tens_size[1] && in.size(2) == tens_size[2]);
  
  autoView(in_v,in,DeviceRead);
  FloatType const* from_base = in_v.data();
  for(int edge_idx=0; edge_idx < nedge; edge_idx++){
    Edge<FloatType> &edge = out.edges[edge_idx];
    Node<FloatType> &send_node = out.nodes[ edge.send_node ];
    Node<FloatType> &recv_node = out.nodes[ edge.recv_node ];

    //stack attributes of send node
    from_base = unstackAttrAdd(send_node, from_base);

    //then receive node      
    from_base = unstackAttrAdd(recv_node, from_base);

    //then edge
    from_base = unstackAttrAdd(edge, from_base);

    //then global
    from_base = unstackAttrAdd(out.global, from_base);
  }
}

template<typename Config>
std::array<int,3> ExtractEdgeUpdateInputComponent<Config>::outputTensorSize(const GraphInitialize &ginit){
  int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
  int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
  int global_attr_size_total = ginit.totalAttribSize(ginit.global_attr_sizes);
  
  std::array<int,3> out;    
  out[0] = ginit.edge_map.size();
  out[1] = 2*node_attr_size_total + edge_attr_size_total + global_attr_size_total;
  out[2] = ginit.batch_size;
  return out;
}


template<typename Config>
Graph<typename Config::FloatType> InsertEdgeUpdateOutputComponent<Config>::value(const Graph<FloatType> &in, const Tensor<FloatType,3> &edge_attr_update){
  if(!setup){
    ginit = in.getInitializer();
    tens_size[0] = in.edges.size();
    tens_size[1] = in.edges[0].totalAttribSize();
    tens_size[2] = ginit.batch_size;
    setup = true;
  }
  assert(edge_attr_update.size(0) == tens_size[0] && edge_attr_update.size(1) == tens_size[1] && edge_attr_update.size(2) == tens_size[2]);
        
  Graph<FloatType> out(in);
    
  autoView(e_v,edge_attr_update,DeviceRead);
    
  FloatType const* from_base = e_v.data();
  for(auto &edge : out.edges) from_base = unstackAttr(edge, from_base);

  return out;
}
  
template<typename Config>
void InsertEdgeUpdateOutputComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,3> &dCost_by_dIn_tens) const{
  Graph<FloatType> in(std::move(_dCost_by_dOut));
    
  Tensor<FloatType,3> &out = dCost_by_dIn_tens;

  assert(setup);
  out = Tensor<FloatType,3>(tens_size);    
  //assert(bool(in.getInitializer() == ginit));
  
  autoView(out_v,out,DeviceWrite);
  FloatType * to_base = out_v.data();
  for(auto const &edge : in.edges) to_base = stackAttr(to_base, edge);
  
  //the contribution to the edges is overwritten from the input graph, but not to the other components
  dCost_by_dIn_graph = Graph<FloatType>(std::move(in));
  for(auto &edge : dCost_by_dIn_graph.edges) edge.setZero();
}

template<typename Config>
std::array<int,3> InsertEdgeUpdateOutputComponent<Config>::inputTensorSize(const GraphInitialize &ginit){
  int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
    
  std::array<int,3> out;    
  out[0] = ginit.edge_map.size();
  out[1] = edge_attr_size_total;
  out[2] = ginit.batch_size;
  return out;
}
