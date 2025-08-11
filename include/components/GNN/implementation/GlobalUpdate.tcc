template<typename Config>
Tensor<typename Config::FloatType,2> ExtractGlobalUpdateInputComponent<Config>::value(const Graph<FloatType> &gaggr){
  assert(gaggr.edges.size() == 1 && gaggr.edges[0].send_node == -1 && gaggr.edges[0].recv_node == -1);
  assert(gaggr.nodes.size() == 1);    
    
  if(!setup){
    ginit = gaggr.getInitializer();
    n_node_attr = gaggr.nodes[0].attributes.size();
    n_edge_attr = gaggr.edges[0].attributes.size();
      
    node_attr_size_total =0;
    for(auto &attr : gaggr.nodes[0].attributes)
      node_attr_size_total += attr.size(0);

    edge_attr_size_total =0;
    for(auto &attr : gaggr.edges[0].attributes)
      edge_attr_size_total += attr.size(0);
      
    tens_size[0] = node_attr_size_total + edge_attr_size_total + ginit.global_attr_size;
    tens_size[1] = ginit.batch_size;

    setup = true;
  }
    
  Tensor<FloatType,2> out(tens_size);
  autoView(out_v,out,DeviceWrite);

  FloatType *to_base = out_v.data();
  for(int a=0;a<n_node_attr;a++)
    to_base = stackAttr(to_base, gaggr.nodes[0].attributes[a]);

  for(int a=0;a<n_edge_attr;a++)
    to_base = stackAttr(to_base, gaggr.edges[0].attributes[a]);

  to_base = stackAttr(to_base, gaggr.global);

  return out;
}

template<typename Config>
void ExtractGlobalUpdateInputComponent<Config>::deriv(Tensor<FloatType,2> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  Tensor<FloatType,2> in(_dCost_by_dOut);
  Graph<FloatType> &out = dCost_by_dIn;

  assert(setup);
  out = Graph<FloatType>(ginit);
    
  assert(in.size(0) == tens_size[0] && in.size(1) == tens_size[1]);
  
  autoView(in_v,in,DeviceRead);
  FloatType const* from_base = in_v.data();

  Edge<FloatType> &agg_edge = out.edges[0];
  Node<FloatType> &agg_node = out.nodes[0];

  for(int a=0;a<n_node_attr;a++)
    from_base = unstackAttrAdd(agg_node.attributes[a], from_base);

  for(int a=0;a<n_edge_attr;a++)
    from_base = unstackAttrAdd(agg_edge.attributes[a], from_base);

  from_base = unstackAttrAdd(out.global, from_base);
}

template<typename Config>
std::array<int,2> ExtractGlobalUpdateInputComponent<Config>::outputTensorSize(const GraphInitialize &ginit){
  int node_attr_size_total = std::accumulate(ginit.node_attr_sizes.begin(),ginit.node_attr_sizes.end(),0);
  int edge_attr_size_total = std::accumulate(ginit.edge_attr_sizes.begin(),ginit.edge_attr_sizes.end(),0);
    
  std::array<int,2> out;    
  out[0] = node_attr_size_total + edge_attr_size_total + ginit.global_attr_size;
  out[1] = ginit.batch_size;
  return out;
}



template<typename Config>
Graph<typename Config::FloatType> InsertGlobalUpdateOutputComponent<Config>::value(const Graph<FloatType> &in, const Tensor<FloatType,2> &global_attr_update){
  if(!setup){
    ginit = in.getInitializer();
    setup = true;
  }
  assert(global_attr_update.size(0) == ginit.global_attr_size && global_attr_update.size(1) == ginit.batch_size);
        
  Graph<FloatType> out(in);
  out.global = global_attr_update;
  return out;
}

template<typename Config>
void InsertGlobalUpdateOutputComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,2> &dCost_by_dIn_tens) const{
  Graph<FloatType> in(std::move(_dCost_by_dOut));
    
  dCost_by_dIn_tens = in.global;
    
  //the contribution to the global attr is overwritten from the input graph, but not to the other components
  dCost_by_dIn_graph = Graph<FloatType>(std::move(in));
  autoView(attr_v, dCost_by_dIn_graph.global, DeviceWrite);
  acceleratorMemSet(attr_v.data(),0,attr_v.data_len() * sizeof(FloatType));
}

template<typename Config>
std::array<int,2> InsertGlobalUpdateOutputComponent<Config>::inputTensorSize(const GraphInitialize &ginit){
  std::array<int,2> out;    
  out[0] = ginit.global_attr_size;
  out[1] = ginit.batch_size;
  return out;
}

