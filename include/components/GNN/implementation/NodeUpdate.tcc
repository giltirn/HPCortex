template<typename Config>
Tensor<typename Config::FloatType,3> ExtractNodeUpdateInputComponent<Config>::value(const Graph<FloatType> &in){
  if(!setup){
    ginit = in.getInitializer();
    n_node_attr = ginit.node_attr_sizes.size();
    n_edge_attr = ginit.edge_attr_sizes.size();
    node_attr_size_total = std::accumulate(ginit.node_attr_sizes.begin(),ginit.node_attr_sizes.end(),0);
    edge_attr_size_total = std::accumulate(ginit.edge_attr_sizes.begin(),ginit.edge_attr_sizes.end(),0);
    
    tens_size[0] = ginit.nnode;
    tens_size[1] = node_attr_size_total + edge_attr_size_total + ginit.global_attr_size;
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
    for(int a=0;a<n_node_attr;a++)
      to_base = stackAttr(to_base, node.attributes[a]);
           
    //then aggregate edge
    for(int a=0;a<n_edge_attr;a++)
      to_base = stackAttr(to_base, agg_edge.attributes[a]);

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

    for(int a=0;a<n_node_attr;a++)
      from_base = unstackAttrAdd(node.attributes[a], from_base);

    for(int a=0;a<n_edge_attr;a++)
      from_base = unstackAttrAdd(agg_edge.attributes[a], from_base);

    from_base = unstackAttrAdd(out.global, from_base);
  }
}

template<typename Config>
std::array<int,3> ExtractNodeUpdateInputComponent<Config>::outputTensorSize(const GraphInitialize &ginit){
  int node_attr_size_total = std::accumulate(ginit.node_attr_sizes.begin(),ginit.node_attr_sizes.end(),0);
  int edge_attr_size_total = std::accumulate(ginit.edge_attr_sizes.begin(),ginit.edge_attr_sizes.end(),0);
    
  std::array<int,3> out;    
  out[0] = ginit.nnode;
  out[1] = node_attr_size_total + edge_attr_size_total + ginit.global_attr_size;
  out[2] = ginit.batch_size;
  return out;
}


template<typename Config>
Graph<typename Config::FloatType> InsertNodeUpdateOutputComponent<Config>::value(const Graph<FloatType> &in, const Tensor<FloatType,3> &node_attr_update){
  if(!setup){
    ginit = in.getInitializer();
    nnode = ginit.nnode;
    n_node_attr = ginit.node_attr_sizes.size();
    node_attr_size_total = std::accumulate(ginit.node_attr_sizes.begin(),ginit.node_attr_sizes.end(),0);
    tens_size[0] = nnode;
    tens_size[1] = node_attr_size_total;
    tens_size[2] = ginit.batch_size;
    setup = true;
  }
  assert(node_attr_update.size(0) == tens_size[0] && node_attr_update.size(1) == tens_size[1] && node_attr_update.size(2) == tens_size[2]);
        
  Graph<FloatType> out(in);    
  autoView(n_v,node_attr_update,DeviceRead);
    
  FloatType const* from_base = n_v.data();
  for(int node_idx=0; node_idx < nnode; node_idx++){
    Node<FloatType> &node = out.nodes[node_idx];
    for(int a=0;a<n_node_attr;a++)
      from_base = unstackAttr(node.attributes[a], from_base);
  }
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
  for(int node_idx=0; node_idx < nnode; node_idx++){
    const Node<FloatType> &node = in.nodes[node_idx];
    for(int a=0;a<n_node_attr;a++)
      to_base = stackAttr(to_base, node.attributes[a]);
  }

  //the contribution to the nodes is overwritten from the input graph, but not to the other components
  dCost_by_dIn_graph = Graph<FloatType>(std::move(in));
  for(auto &node : dCost_by_dIn_graph.nodes)
    for(auto &attr : node.attributes){
      autoView(attr_v, attr, DeviceWrite);
      acceleratorMemSet(attr_v.data(),0,attr_v.data_len() * sizeof(FloatType));
    }   
}


  
template<typename Config>
std::array<int,3> InsertNodeUpdateOutputComponent<Config>::inputTensorSize(const GraphInitialize &ginit){
  int node_attr_size_total = std::accumulate(ginit.node_attr_sizes.begin(),ginit.node_attr_sizes.end(),0);
    
  std::array<int,3> out;    
  out[0] = ginit.nnode;
  out[1] = node_attr_size_total;
  out[2] = ginit.batch_size;
  return out;
}


