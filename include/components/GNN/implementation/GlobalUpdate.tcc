template<typename Config>
Tensor<typename Config::FloatType,2> ExtractGlobalUpdateInputComponent<Config>::value(const Graph<FloatType> &gaggr){
  assert(gaggr.edges.nElem() == 1 && gaggr.edges.sendNode(0) == -1 && gaggr.edges.recvNode(0) == -1);
  assert(gaggr.nodes.nElem() == 1);    
    
  if(!setup){
    ginit = gaggr.getInitializer();
    int node_attr_size_total = ginit.totalAttribSize(ginit.node_attr_sizes);
    int edge_attr_size_total = ginit.totalAttribSize(ginit.edge_attr_sizes);
    int global_attr_size_total =  ginit.totalAttribSize(ginit.global_attr_sizes);
      
    tens_size[0] = node_attr_size_total + edge_attr_size_total + global_attr_size_total;
    tens_size[1] = ginit.batch_size;

    int nedge_attr = ginit.edge_attr_sizes.size();
    int nnode_attr = ginit.node_attr_sizes.size();
    int nglob_attr = ginit.global_attr_sizes.size();
    int ncopy = nnode_attr + nedge_attr + nglob_attr;
    
    copy_template = Vector<elemCopyTemplate>(ncopy, MemoryManager::Pool::HostPool);
    {
      autoView(cpt, copy_template, HostWrite);
      int i=0;
      int off = 0;
      for(int na=0;na<nnode_attr;na++){ //stack agg. node attribs
	int attr_sz = ginit.node_attr_sizes[na];
	cpt(i++) = elemCopyTemplate(GraphElementType::Nodes, na, 0, off); 
	off += attr_sz;
      }
      for(int ea=0;ea<nedge_attr;ea++){ //stack agg. edge attribs
	int attr_sz = ginit.edge_attr_sizes[ea];
	cpt(i++) = elemCopyTemplate(GraphElementType::Edges, ea, 0, off); 
	off += attr_sz;
      }
      for(int ng=0;ng<nglob_attr;ng++){ //stack global attribs
	int attr_sz = ginit.global_attr_sizes[ng];
	cpt(i++) = elemCopyTemplate(GraphElementType::Global, ng, 0, off);
	off += attr_sz;
      }
    }
      
    setup = true;
  }
    
  Tensor<FloatType,2> out(tens_size);
  stackAttr(out, gaggr, copy_template);
  return out;
}

template<typename Config>
void ExtractGlobalUpdateInputComponent<Config>::deriv(Tensor<FloatType,2> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn) const{
  Tensor<FloatType,2> dCost_by_dOut(_dCost_by_dOut);
  assert(setup);
  dCost_by_dIn = Graph<FloatType>(ginit);
    
  assert(dCost_by_dOut.size(0) == tens_size[0] && dCost_by_dOut.size(1) == tens_size[1]);
  unstackAttrAdd(dCost_by_dIn, dCost_by_dOut, copy_template);
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
template<typename InputGraphType, enable_if_fwd_ref<InputGraphType, Graph<typename Config::FloatType> > >
Graph<typename Config::FloatType> InsertGlobalUpdateOutputComponent<Config>::value(InputGraphType &&in, const Tensor<FloatType,2> &global_attr_update){
  if(!setup){
    ginit = in.getInitializer();
    global_attr_size_total =  ginit.totalAttribSize(ginit.global_attr_sizes);
    setup = true;
  }
  assert(global_attr_update.size(0) == global_attr_size_total && global_attr_update.size(1) == ginit.batch_size);
        
  Graph<FloatType> out(std::forward<InputGraphType>(in));
  unstackAttrSingleElem(out.global, global_attr_update);
  return out;
}

template<typename Config>
void InsertGlobalUpdateOutputComponent<Config>::deriv(Graph<FloatType> &&_dCost_by_dOut, Graph<FloatType> &dCost_by_dIn_graph, Tensor<FloatType,2> &dCost_by_dIn_tens) const{
  Graph<FloatType> dCost_by_dOut(std::move(_dCost_by_dOut));

  dCost_by_dIn_tens = Tensor<FloatType,2>(global_attr_size_total, ginit.batch_size, 0.);
  stackAttrSingleElem(dCost_by_dIn_tens, dCost_by_dOut.global);
      
  //the contribution to the global attr is overwritten from the input graph, but not to the other components
  dCost_by_dIn_graph = Graph<FloatType>(std::move(dCost_by_dOut));
  dCost_by_dIn_graph.global.setZero();
}

template<typename Config>
std::array<int,2> InsertGlobalUpdateOutputComponent<Config>::inputTensorSize(const GraphInitialize &ginit){
  std::array<int,2> out;    
  out[0] = ginit.totalAttribSize(ginit.global_attr_sizes);
  out[1] = ginit.batch_size;
  return out;
}

