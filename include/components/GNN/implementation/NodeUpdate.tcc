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

    int nedge_attr = ginit.edge_attr_sizes.size();
    int nnode_attr = ginit.node_attr_sizes.size();
    int nglob_attr = ginit.global_attr_sizes.size();
    int ncopy_per_node = nnode_attr + nedge_attr + nglob_attr;
    
    copy_template = Matrix<elemCopyTemplate>(ginit.nnode, ncopy_per_node, MemoryManager::Pool::HostPool);
    {
      autoView(cpt, copy_template, HostWrite);
      for(int node_idx=0;node_idx<ginit.nnode;node_idx++){
	int i=0;
	int off = 0;
	for(int na=0;na<nnode_attr;na++){ //stack self attribs
	  int attr_sz = ginit.node_attr_sizes[na];
	  cpt(node_idx,i++) = elemCopyTemplate(GraphElementType::Nodes, na, node_idx, off); 
	  off += attr_sz;
	}
	for(int ea=0;ea<nedge_attr;ea++){ //stack agg. edge attribs
	  int attr_sz = ginit.edge_attr_sizes[ea];
	  cpt(node_idx,i++) = elemCopyTemplate(GraphElementType::Edges, ea, node_idx, off); 
	  off += attr_sz;
	}
	for(int ng=0;ng<nglob_attr;ng++){ //stack global attribs
	  int attr_sz = ginit.global_attr_sizes[ng];
	  cpt(node_idx,i++) = elemCopyTemplate(GraphElementType::Global, ng, 0, off);
	  off += attr_sz;
	}
      }
    }

    
    setup = true;
  }
  if(ginit.batch_size != in.nodes.batchSize())
    ginit.batch_size = tens_size[2] = in.nodes.batchSize();

  Tensor<FloatType,3> out(tens_size);
  stackAttr(out, in, copy_template);

  return out;
}

template<typename Config>	
void ExtractNodeUpdateInputComponent<Config>::deriv(Tensor<FloatType,3> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  Tensor<FloatType,3> dCost_by_dOut(_dCost_by_dOut);
  assert(setup);
  assert(dCost_by_dOut.size(0) == tens_size[0] && dCost_by_dOut.size(1) == tens_size[1] && dCost_by_dOut.size(2) == tens_size[2]);
  dCost_by_dIn = Graph<FloatType>(ginit);
  unstackAttrAdd(dCost_by_dIn, dCost_by_dOut, copy_template);
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
template<typename InputGraphType, enable_if_fwd_ref<InputGraphType, Graph<typename Config::FloatType> > >
Graph<typename Config::FloatType> InsertNodeUpdateOutputComponent<Config>::value(InputGraphType &&in, const Tensor<FloatType,3> &node_attr_update){
  if(!setup){
    ginit = in.getInitializer();
    int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
    tens_size[0] = ginit.nnode;
    tens_size[1] = node_attr_size_total;
    tens_size[2] = ginit.batch_size;
    setup = true;
  }
  if(ginit.batch_size != in.nodes.batchSize())
    ginit.batch_size = tens_size[2] = in.nodes.batchSize();
  
  assert(node_attr_update.size(0) == tens_size[0] && node_attr_update.size(1) == tens_size[1] && node_attr_update.size(2) == tens_size[2]);
        
  Graph<FloatType> out(std::forward<InputGraphType>(in));    
  autoView(n_v,node_attr_update,DeviceRead);
  unstackAttr(out.nodes, node_attr_update);
  return out;
}

template<typename Config>
void InsertNodeUpdateOutputComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,3> &dCost_by_dIn_tens) const{
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));    
  assert(setup);
  dCost_by_dIn_tens = Tensor<FloatType,3>(tens_size);    
  assert(bool(dCost_by_dOut.getInitializer() == ginit));

  stackAttr(dCost_by_dIn_tens, dCost_by_dOut.nodes);

  //the contribution to the nodes is overwritten from the input graph, but not to the other components
  dCost_by_dIn_graph = Graph<FloatType>(std::move(dCost_by_dOut));
  dCost_by_dIn_graph.nodes.setZero();
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
