template<typename FloatType>
bool equal(const AttributedGraphElements<FloatType> &a, const AttributedGraphElements<FloatType> &b, bool verbose = true){
  if(a.attributes.size() != b.attributes.size()) return false;
  for(int aa=0;aa<a.attributes.size();aa++)    
    if(!equal(a.attributes[aa],b.attributes[aa],verbose)){
      if(verbose) std::cout << "Failed on attrib " << aa << std::endl;
      return false;
    }
  return true;
}

template<typename FloatType>
bool equal(const Edges<FloatType> &a, const Edges<FloatType> &b, bool verbose = true){
  if(a.attributes.size() != b.attributes.size()){
    if(verbose) std::cout << "Attributes size differs" << std::endl;
    return false;
  }
  if(a.edge_map != b.edge_map){
    if(verbose) std::cout << "Edge map differs" << std::endl;
    return false;
  }
  for(int aa=0;aa<a.attributes.size();aa++)    
    if(!equal(a.attributes[aa],b.attributes[aa],verbose)){
      if(verbose) std::cout << "Failed on attrib " << aa << std::endl;
      return false;
    }
  return true;
}

template<typename FloatType>
bool equal(const Graph<FloatType> &g1, const Graph<FloatType> &g2, bool verbose=true){
  if(!equal(g1.nodes,g2.nodes,verbose)){
    if(verbose) std::cout << "Failed on nodes" << std::endl;
    return false;
  }
  if(!equal(g1.edges,g2.edges,verbose)){
    std::cout << "Failed on edges" << std::endl;
    if(verbose) return false;
  }
  if(!equal(g1.global,g2.global,verbose)){
    if(verbose) std::cout << "Failed on global" << std::endl;
    return false;
  }
  return true;
}


template<typename FloatType>
bool abs_near(const AttributedGraphElements<FloatType> &a, const AttributedGraphElements<FloatType> &b, FloatType delta, bool verbose = true){ 
  if(a.attributes.size() != b.attributes.size()) return false;
  for(int aa=0;aa<a.attributes.size();aa++)    
    if(!abs_near(a.attributes[aa],b.attributes[aa],delta,verbose)){
      if(verbose) std::cout << "Failed on attrib " << aa << std::endl;
      return false;
    }
  return true;
}

template<typename FloatType>
bool abs_near(const Edges<FloatType> &a, const Edges<FloatType> &b, FloatType delta, bool verbose = true){
  if(a.attributes.size() != b.attributes.size()) return false;
  if(a.edge_map != b.edge_map) return false;
  for(int aa=0;aa<a.attributes.size();aa++)    
    if(!abs_near(a.attributes[aa],b.attributes[aa],delta,verbose)){
      if(verbose) std::cout << "Failed on attrib " << aa << std::endl;
      return false;
    }
  return true;
}

template<typename FloatType>
bool abs_near(const Graph<FloatType> &g1, const Graph<FloatType> &g2, FloatType delta, bool verbose=true){
  if(!abs_near(g1.nodes,g2.nodes,delta,verbose)){
    if(verbose) std::cout << "Failed on nodes" << std::endl;
    return false;
  }
  if(!abs_near(g1.edges,g2.edges,delta,verbose)){
    std::cout << "Failed on edges" << std::endl;
    if(verbose) return false;
  }
  if(!abs_near(g1.global,g2.global,delta,verbose)){
    if(verbose) std::cout << "Failed on global" << std::endl;
    return false;
  }
  return true;
}

template<typename Config, typename Layer>
struct GraphInGraphOutLayerWrapper{
  EXTRACT_CONFIG_TYPES;
  
  Layer &layer;
  GraphInitialize ginit;
  int graph_flat_size;
  
  GraphInGraphOutLayerWrapper(Layer &layer, const GraphInitialize &ginit, const int graph_flat_size): layer(layer), ginit(ginit), graph_flat_size(graph_flat_size){}

  size_t outputLinearSize() const{ return graph_flat_size; }
  size_t inputLinearSize() const{ return graph_flat_size; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    ginit.batch_size = in.size(1);
    Graph<FloatType> ing = unflattenFromBatchVector(in, ginit);
    return flattenToBatchVector(layer.value(ing, enable_deriv) );
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    ginit.batch_size = _above_deriv_lin.size(1);
    Graph<FloatType> above_deriv = unflattenFromBatchVector(_above_deriv_lin, ginit);
    Graph<FloatType> cost_deriv_ing;
    layer.deriv(cost_deriv_params, off, std::move(above_deriv), &cost_deriv_ing);
    cost_deriv_inputs = flattenToBatchVector(cost_deriv_ing);
  }
    
  void update(int off, const Vector<FloatType> &new_params){
    layer.update(off,new_params);
  }
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){
    layer.step(off,derivs,eps);
  }
  inline int nparams() const{ return layer.nparams(); }

  void getParams(Vector<FloatType> &into, int off){
    layer.getParams(into,off);
  }

  std::string inCoord(size_t i, int b, int batch_size) const{
    return "";
  }      
};

void testGraph(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  int batch_size=4;
  
  GraphInitialize ginit;
  ginit.nnode = 3;
  ginit.node_attr_sizes = std::vector<int>({3,4});
  ginit.edge_attr_sizes = std::vector<int>({5,6});
  //edges in circle
  ginit.edge_map = std::vector<std::pair<int,int> >({  {0,1}, {1,2}, {2,0}, {1,0}, {2,1}, {0,2} });
  ginit.global_attr_sizes = std::vector<int>({2});
  ginit.batch_size = batch_size;

  //test insertCompleteBatch
  {
    GraphInitialize ginit_unbatched(ginit);
    ginit_unbatched.batch_size=1;
    
    std::vector<Graph<FloatType> > bgraphs(batch_size, ginit_unbatched);
    for(int b=0;b<batch_size;b++) bgraphs[b].applyToAllAttributes([&](Tensor<FloatType,3> &m){ uniformRandom(m,rng); });

    Graph<FloatType> graph(ginit);
    std::vector<Graph<FloatType> const*> gptrs(batch_size);
    for(int b=0;b<batch_size;b++)
      gptrs[b] = &bgraphs[b];
    graph.insertCompleteBatch(gptrs.data());
    
    
    {
      for(int a=0;a<ginit.node_attr_sizes.size();a++){
	autoView(a_v, graph.nodes.attributes[a], HostRead);
	for(int b=0;b<batch_size;b++){
	  autoView(fa_v, bgraphs[b].nodes.attributes[a], HostRead);
	  assert(fa_v.size(0) == a_v.size(0) && fa_v.size(1) == a_v.size(1));
	  for(int n=0;n<ginit.nnode;n++){	    
	    for(int i=0;i<fa_v.size(1);i++)
	      assert(fa_v(n,i,0) == a_v(n,i,b));
	  }
	}
      }
    }
    {
      for(int a=0;a<ginit.node_attr_sizes.size();a++){
	autoView(a_v, graph.edges.attributes[a], HostRead);
	for(int b=0;b<batch_size;b++){
	  autoView(fa_v, bgraphs[b].edges.attributes[a], HostRead);
	  assert(fa_v.size(0) == a_v.size(0) && fa_v.size(1) == a_v.size(1));
	  for(int e=0;e<ginit.edge_map.size();e++){	    
	    for(int i=0;i<fa_v.size(1);i++)
	      assert(fa_v(e,i,0) == a_v(e,i,b));
	  }
	}
      }
    }
    
    for(int a=0;a<ginit.global_attr_sizes.size();a++){
      autoView(a_v, graph.global.attributes[a], HostRead);
      for(int b=0;b<batch_size;b++){
	autoView(fa_v, bgraphs[b].global.attributes[a], HostRead);
	assert(fa_v.size(0) == a_v.size(0));
	for(int i=0;i<fa_v.size(0);i++)
	  assert(fa_v(0,i,0) == a_v(0,i,b));
      }
    }    
  }

  std::cout << "testGraph passed" << std::endl;
}
