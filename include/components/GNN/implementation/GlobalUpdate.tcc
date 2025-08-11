template<typename Config>
Tensor<typename Config::FloatType,2> ExtractGlobalUpdateInputComponent<Config>::value(const Graph<FloatType> &gaggr){
  assert(gaggr.edges.size() == 1 && gaggr.edges[0].send_node == -1 && gaggr.edges[0].recv_node == -1);
  assert(gaggr.nodes.size() == 1);    
    
  if(!setup){
    ginit = gaggr.getInitializer();
    int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
    int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
    int global_attr_size_total =  ginit.totalAttribSize(ginit.global_attr_sizes);
      
    tens_size[0] = node_attr_size_total + edge_attr_size_total + global_attr_size_total;
    tens_size[1] = ginit.batch_size;

    setup = true;
  }
    
  Tensor<FloatType,2> out(tens_size);
  autoView(out_v,out,DeviceWrite);

  FloatType *to_base = out_v.data();
  to_base = stackAttr(to_base, gaggr.nodes[0]);
  to_base = stackAttr(to_base, gaggr.edges[0]);    
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

  from_base = unstackAttrAdd(out.nodes[0], from_base);
  from_base = unstackAttrAdd(out.edges[0], from_base);
  from_base = unstackAttrAdd(out.global, from_base);
}

template<typename Config>
std::array<int,2> ExtractGlobalUpdateInputComponent<Config>::outputTensorSize(const GraphInitialize &ginit){
  int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
  int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
  int global_attr_size_total =  ginit.totalAttribSize(ginit.global_attr_sizes);
    
  std::array<int,2> out;    
  out[0] = node_attr_size_total + edge_attr_size_total + global_attr_size_total;
  out[1] = ginit.batch_size;
  return out;
}



template<typename Config>
Graph<typename Config::FloatType> InsertGlobalUpdateOutputComponent<Config>::value(const Graph<FloatType> &in, const Tensor<FloatType,2> &global_attr_update){
  if(!setup){
    ginit = in.getInitializer();
    global_attr_size_total =  ginit.totalAttribSize(ginit.global_attr_sizes);
    setup = true;
  }
  assert(global_attr_update.size(0) == global_attr_size_total && global_attr_update.size(1) == ginit.batch_size);
        
  Graph<FloatType> out(in);
  autoView(t_v, global_attr_update, DeviceRead);  
  FloatType const *from_base = t_v.data();
  from_base = unstackAttr(out.global, from_base);
  return out;
}

template<typename Config>
void InsertGlobalUpdateOutputComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,2> &dCost_by_dIn_tens) const{
  Graph<FloatType> in(std::move(_dCost_by_dOut));

  dCost_by_dIn_tens = Tensor<FloatType,2>(global_attr_size_total, ginit.batch_size, 0.);
  
  autoView(out_v,dCost_by_dIn_tens,DeviceWrite);
  FloatType * to_base = out_v.data();
  to_base = stackAttr(to_base, in.global);  
    
  //the contribution to the global attr is overwritten from the input graph, but not to the other components
  dCost_by_dIn_graph = Graph<FloatType>(std::move(in));
  dCost_by_dIn_graph.global.setZero();
}

template<typename Config>
std::array<int,2> InsertGlobalUpdateOutputComponent<Config>::inputTensorSize(const GraphInitialize &ginit){
  std::array<int,2> out;    
  out[0] = ginit.totalAttribSize(ginit.global_attr_sizes);
  out[1] = ginit.batch_size;
  return out;
}

