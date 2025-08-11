template<typename Config>
Tensor<typename Config::FloatType,3> ExtractNodeUpdateInputComponent<Config>::value(const Graph<FloatType> &in){
  if(!setup){
    ginit = in.getInitializer();
    int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
    int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
    int global_attr_size_total = ginit.totalAttribSize(ginit.global_attr_sizes);
        
    tens_size[0] = ginit.nnode;
    tens_size[1] = node_attr_size_total + edge_attr_size_total + global_attr_size_total;
    tens_size[2] = ginit.batch_size;

    setup = true;
  }

  Tensor<FloatType,3> out(tens_size);
  autoView(out_v,out,DeviceWrite);

  FloatType *to_base = out_v.data();
  for(int node_idx=0; node_idx < ginit.nnode; node_idx++){
    const Node<FloatType> &node = in.nodes[node_idx];
    const Edge<FloatType> &agg_edge = in.edges[node_idx];
    assert(agg_edge.recv_node == node_idx && agg_edge.send_node == -1);
      
    //stack attributes of node
    to_base = stackAttr(to_base, node);
               
    //then aggregate edge
    to_base = stackAttr(to_base, agg_edge);

    //then global
    to_base = stackAttr(to_base, in.global);
  }
  return out;
}

template<typename Config>	
void ExtractNodeUpdateInputComponent<Config>::deriv(Tensor<FloatType,3> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  Tensor<FloatType,3> in(_dCost_by_dOut);
  Graph<FloatType> &out = dCost_by_dIn;

  assert(setup);
  out = Graph<FloatType>(ginit);
    
  assert(in.size(0) == tens_size[0] && in.size(1) == tens_size[1] && in.size(2) == tens_size[2]);
  
  autoView(in_v,in,DeviceRead);
  FloatType const* from_base = in_v.data();
  for(int node_idx=0; node_idx < ginit.nnode; node_idx++){
    Edge<FloatType> &agg_edge = out.edges[node_idx];
    Node<FloatType> &node = out.nodes[node_idx];

    from_base = unstackAttrAdd(node, from_base);
    from_base = unstackAttrAdd(agg_edge, from_base);
    from_base = unstackAttrAdd(out.global, from_base);
  }
}

template<typename Config>
std::array<int,3> ExtractNodeUpdateInputComponent<Config>::outputTensorSize(const GraphInitialize &ginit){
  int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
  int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
  int global_attr_size_total = ginit.totalAttribSize(ginit.global_attr_sizes);
     
  std::array<int,3> out;    
  out[0] = ginit.nnode;
  out[1] = node_attr_size_total + edge_attr_size_total + global_attr_size_total;
  out[2] = ginit.batch_size;
  return out;
}


template<typename Config>
Graph<typename Config::FloatType> InsertNodeUpdateOutputComponent<Config>::value(const Graph<FloatType> &in, const Tensor<FloatType,3> &node_attr_update){
  if(!setup){
    ginit = in.getInitializer();
    int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
    tens_size[0] = ginit.nnode;
    tens_size[1] = node_attr_size_total;
    tens_size[2] = ginit.batch_size;
    setup = true;
  }
  assert(node_attr_update.size(0) == tens_size[0] && node_attr_update.size(1) == tens_size[1] && node_attr_update.size(2) == tens_size[2]);
        
  Graph<FloatType> out(in);    
  autoView(n_v,node_attr_update,DeviceRead);
    
  FloatType const* from_base = n_v.data();
  for(auto &n : out.nodes) from_base = unstackAttr(n, from_base);
  return out;
}

template<typename Config>
void InsertNodeUpdateOutputComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,3> &dCost_by_dIn_tens) const{
  Graph<FloatType> in(std::move(_dCost_by_dOut));
    
  Tensor<FloatType,3> &out = dCost_by_dIn_tens;

  assert(setup);
  out = Tensor<FloatType,3>(tens_size);    
  assert(bool(in.getInitializer() == ginit));
  
  autoView(out_v,out,DeviceWrite);
  FloatType * to_base = out_v.data();
  for(auto const &n : in.nodes) to_base = stackAttr(to_base, n);

  //the contribution to the nodes is overwritten from the input graph, but not to the other components
  dCost_by_dIn_graph = Graph<FloatType>(std::move(in));
  for(auto &node : dCost_by_dIn_graph.nodes) node.setZero();
}


  
template<typename Config>
std::array<int,3> InsertNodeUpdateOutputComponent<Config>::inputTensorSize(const GraphInitialize &ginit){
  int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
    
  std::array<int,3> out;    
  out[0] = ginit.nnode;
  out[1] = node_attr_size_total;
  out[2] = ginit.batch_size;
  return out;
}


