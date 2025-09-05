template<typename FloatType>
bool equal(const AttributedGraphElement<FloatType> &a, const AttributedGraphElement<FloatType> &b, bool verbose = true){
  if(a.attributes.size() != b.attributes.size()) return false;
  for(int aa=0;aa<a.attributes.size();aa++)    
    if(!equal(a.attributes[aa],b.attributes[aa],verbose)){
      if(verbose) std::cout << "Failed on attrib " << aa << std::endl;
      return false;
    }
  return true;
}

template<typename FloatType>
bool equal(const Edge<FloatType> &a, const Edge<FloatType> &b, bool verbose = true){
  if(a.attributes.size() != b.attributes.size()) return false;
  if(a.send_node != b.send_node) return false;
  if(a.recv_node != b.recv_node) return false;    
  for(int aa=0;aa<a.attributes.size();aa++)    
    if(!equal(a.attributes[aa],b.attributes[aa],verbose)){
      if(verbose) std::cout << "Failed on attrib " << aa << std::endl;
      return false;
    }
  return true;
}

template<typename FloatType>
bool equal(const Graph<FloatType> &g1, const Graph<FloatType> &g2, bool verbose=true){
  if(g1.nodes.size() != g2.nodes.size()) return false;
  if(g1.edges.size() != g2.edges.size()) return false;
  for(int n=0;n<g1.nodes.size();n++){
    if(g1.nodes[n].attributes.size() != g2.nodes[n].attributes.size()) return false;
    for(int a=0;a<g1.nodes[n].attributes.size();a++)    
      if(!equal(g1.nodes[n].attributes[a],g2.nodes[n].attributes[a],verbose)){
	if(verbose) std::cout << "Failed on node " << n << " attrib " << a << std::endl;
	return false;
      }
  }
  for(int n=0;n<g1.edges.size();n++){
    if(!equal(g1.edges[n], g2.edges[n], verbose)){
      if(verbose) std::cout << "Failed on edge " << n << std::endl;
      return false;
    }
  }
  if(!equal(g1.global,g2.global,verbose)){
    if(verbose) std::cout << "Failed on global" << std::endl;
    return false;
  }
  return true;
}
template<typename FloatType>
bool abs_near(const Graph<FloatType> &g1, const Graph<FloatType> &g2, FloatType delta, bool verbose=true){
  if(g1.nodes.size() != g2.nodes.size()) return false;
  if(g1.edges.size() != g2.edges.size()) return false;
  for(int n=0;n<g1.nodes.size();n++){
    if(g1.nodes[n].attributes.size() != g2.nodes[n].attributes.size()) return false;
    for(int a=0;a<g1.nodes[n].attributes.size();a++)    
      if(!abs_near(g1.nodes[n].attributes[a],g2.nodes[n].attributes[a],delta,verbose)){
	if(verbose) std::cout << "Failed on node " << n << " attrib " << a << std::endl;
	return false;
      }
  }
  for(int n=0;n<g1.edges.size();n++){
    if(g1.edges[n].attributes.size() != g2.edges[n].attributes.size()) return false;
    if(g1.edges[n].send_node != g2.edges[n].send_node) return false;
    if(g1.edges[n].recv_node != g2.edges[n].recv_node) return false;    
    for(int a=0;a<g1.edges[n].attributes.size();a++)    
      if(!abs_near(g1.edges[n].attributes[a],g2.edges[n].attributes[a],delta,verbose)){
	if(verbose) std::cout << "Failed on edge " << n << " attrib " << a << std::endl;
	return false;
      }
  }

  if(g1.global.attributes.size() != g2.global.attributes.size()) return false;
  for(int a=0;a<g1.global.attributes.size();a++)    
    if(!abs_near(g1.global.attributes[a],g2.global.attributes[a],delta,verbose)){
      if(verbose) std::cout << "Failed on global attrib " << a << std::endl;
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
  
  Vector<FloatType> value(const Vector<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    Graph<FloatType> ing(ginit);
    unflatten(ing, in);   
    return flatten(layer.value(ing, enable_deriv) );
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Graph<FloatType> above_deriv(ginit);
    unflatten(above_deriv, above_deriv_lin);
    Graph<FloatType> cost_deriv_ing;
    layer.deriv(cost_deriv_params, off, std::move(above_deriv), &cost_deriv_ing);
    cost_deriv_inputs = flatten(cost_deriv_ing);
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

  std::string inCoord(size_t i) const{
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

  //test insertBatch, insertCompleteBatch
  {
    GraphInitialize ginit_unbatched(ginit);
    ginit_unbatched.batch_size=1;
    
    std::vector<Graph<FloatType> > bgraphs(batch_size, ginit_unbatched);
    for(int b=0;b<batch_size;b++) bgraphs[b].applyToAllAttributes([&](Matrix<FloatType> &m){ uniformRandom(m,rng); });

    Graph<FloatType> graph(ginit);
    for(int b=0;b<batch_size;b++) graph.insertBatch(bgraphs[b],b);

    for(int n=0;n<ginit.nnode;n++){
      for(int a=0;a<ginit.node_attr_sizes.size();a++){
	autoView(a_v, graph.nodes[n].attributes[a], HostRead);
	for(int b=0;b<batch_size;b++){
	  autoView(fa_v, bgraphs[b].nodes[n].attributes[a], HostRead);
	  assert(fa_v.size(0) == a_v.size(0));
	  for(int i=0;i<fa_v.size(0);i++)
	    assert(fa_v(i,0) == a_v(i,b));
	}
      }
    }
    for(int e=0;e<ginit.edge_map.size();e++){
      for(int a=0;a<ginit.edge_attr_sizes.size();a++){
	autoView(a_v, graph.edges[e].attributes[a], HostRead);
	for(int b=0;b<batch_size;b++){
	  autoView(fa_v, bgraphs[b].edges[e].attributes[a], HostRead);
	  assert(fa_v.size(0) == a_v.size(0));
	  for(int i=0;i<fa_v.size(0);i++)
	    assert(fa_v(i,0) == a_v(i,b));
	}
      }
    }
    for(int a=0;a<ginit.global_attr_sizes.size();a++){
      autoView(a_v, graph.global.attributes[a], HostRead);
      for(int b=0;b<batch_size;b++){
	autoView(fa_v, bgraphs[b].global.attributes[a], HostRead);
	assert(fa_v.size(0) == a_v.size(0));
	for(int i=0;i<fa_v.size(0);i++)
	  assert(fa_v(i,0) == a_v(i,b));
      }
    }

    Graph<FloatType> graph2(ginit);
    std::vector<Graph<FloatType> const*> gptrs(batch_size);
    for(int b=0;b<batch_size;b++) gptrs[b] = &bgraphs[b];    
    graph2.insertCompleteBatch(gptrs.data());

    assert(equal(graph,graph2,true));
    
  }
}
