template<typename Config>
Tensor<typename Config::FloatType,3> ExtractEdgeUpdateInputComponent<Config>::value(const Graph<FloatType> &in){
  if(!setup){
    ginit = in.getInitializer();
    nedge = ginit.edge_map.size();
    int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
    int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
    int global_attr_size_total = ginit.totalAttribSize(ginit.global_attr_sizes);
    
    tens_size[0] = ginit.edge_map.size();
    tens_size[1] = 2*node_attr_size_total + edge_attr_size_total + global_attr_size_total;
    tens_size[2] = ginit.batch_size;

    int nedge_attr = ginit.edge_attr_sizes.size();
    int nnode_attr = ginit.node_attr_sizes.size();
    int nglob_attr = ginit.global_attr_sizes.size();
    int ncopy_per_edge = nedge_attr + 2*nnode_attr + nglob_attr;

    //pointer list will be [edges,nodes,globals]
    //elemCopyTemplate(int gelem_ptr_idx, int gelem_attrib, int gelem_elem, int stacked_offset)
  
    copy_template = Matrix<elemCopyTemplate>(nedge, ncopy_per_edge, MemoryManager::Pool::HostPool);
    {
      autoView(cpt, copy_template, HostWrite);
      for(int edge_idx=0;edge_idx<nedge;edge_idx++){
	int i=0;
	int off = 0;
	for(int ea=0;ea<nedge_attr;ea++){ //stack self attribs
	  int attr_sz = ginit.edge_attr_sizes[ea];
	  cpt(edge_idx,i++) = elemCopyTemplate(GraphElementType::Edges, ea, edge_idx, off); 
	  off += attr_sz;
	}
	
	for(int na=0;na<nnode_attr;na++){ //stack send, recv node attribs
	  int send_node = in.edges.edge_map[edge_idx].first;
	  int recv_node = in.edges.edge_map[edge_idx].second;
	  int attr_sz = ginit.node_attr_sizes[na];
	  cpt(edge_idx,i++) = elemCopyTemplate(GraphElementType::Nodes, na, send_node, off);
	  off += attr_sz;
	  cpt(edge_idx,i++) = elemCopyTemplate(GraphElementType::Nodes, na, recv_node, off);
	  off += attr_sz;
	}
	for(int ng=0;ng<nglob_attr;ng++){
	  int attr_sz = ginit.global_attr_sizes[ng];
	  cpt(edge_idx,i++) = elemCopyTemplate(GraphElementType::Global, ng, 0, off);
	  off += attr_sz;
	}
      }
    }
    setup = true;
  }
  if(in.edges.batchSize() != ginit.batch_size)
    ginit.batch_size = tens_size[2] = in.edges.batchSize();


  Tensor<FloatType,3> out(tens_size); //[edge, flat_attr_idx, batch]
  stackAttr(out, in, copy_template);
  return out;
}
  
template<typename Config>
void ExtractEdgeUpdateInputComponent<Config>::deriv(Tensor<FloatType,3> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  Tensor<FloatType,3> dCost_by_dOut(_dCost_by_dOut);

  assert(setup);
  dCost_by_dIn = Graph<FloatType>(ginit);
    
  assert(dCost_by_dOut.size(0) == tens_size[0] && dCost_by_dOut.size(1) == tens_size[1] && dCost_by_dOut.size(2) == tens_size[2]);
  unstackAttrAdd(dCost_by_dIn, dCost_by_dOut, copy_template);
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
template<typename InputGraphType, enable_if_fwd_ref<InputGraphType, Graph<typename Config::FloatType> > >
Graph<typename Config::FloatType> InsertEdgeUpdateOutputComponent<Config>::value(InputGraphType &&in, const Tensor<FloatType,3> &edge_attr_update){
  if(!setup){
    ginit = in.getInitializer();
    tens_size[0] = in.edges.nElem();
    tens_size[1] = in.edges.totalAttribSize();
    tens_size[2] = ginit.batch_size;
    setup = true;
  }
  if(ginit.batch_size != in.edges.batchSize())
    ginit.batch_size = tens_size[2] = in.edges.batchSize();
  
  assert(edge_attr_update.size(0) == tens_size[0] && edge_attr_update.size(1) == tens_size[1] && edge_attr_update.size(2) == tens_size[2]);
        
  Graph<FloatType> out(std::forward<InputGraphType>(in)); //avoid copy if possible
  unstackAttr(out.edges, edge_attr_update); 
  return out;
}
  
template<typename Config>
void InsertEdgeUpdateOutputComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,3> &dCost_by_dIn_tens) const{
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));
    
  assert(setup);
  dCost_by_dIn_tens = Tensor<FloatType,3>(tens_size);
  stackAttr(dCost_by_dIn_tens, dCost_by_dOut.edges);
  
  //the contribution to the edges is overwritten from the input graph, but not to the other components
  dCost_by_dIn_graph = Graph<FloatType>(std::move(dCost_by_dOut));
  dCost_by_dIn_graph.edges.setZero();
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
