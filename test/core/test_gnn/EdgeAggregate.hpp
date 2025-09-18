template<typename FloatType>
Graph<FloatType> expectEdgeAggregateSum(const Graph<FloatType> &in){
  Graph<FloatType> out(in);
  for(int a=0;a<out.edges.nAttrib();a++)
    out.edges.attributes[a] = Tensor<FloatType,3>(in.nodes.nElem(), in.edges.attribSize(a), in.edges.batchSize());

  autoView(edges_in_v, in.edges.attributes,HostRead);
  autoView(edges_out_v, out.edges.attributes,HostWrite);
  out.edges.edge_map.resize(in.nodes.nElem());
  
  for(int n=0;n<in.nodes.nElem();n++){
    out.edges.edge_map[n].first = -1;
    out.edges.edge_map[n].second = n;

    bool first = true;
    for(int e=0;e<in.edges.nElem();e++)
      if(in.edges.recvNode(e) == n){
	for(int a=0;a<in.edges.nAttrib();a++){
	  for(int i=0;i<in.edges.attribSize(a);i++)
	    for(int b=0;b<in.edges.batchSize();b++)
	      if(first) edges_out_v[a](n,i,b) = edges_in_v[a](e,i,b);
	      else edges_out_v[a](n,i,b) += edges_in_v[a](e,i,b);
	}
	first = false;
      }
  }
  return out;
}

template<typename Config>
struct EdgeAggregateSumComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  EdgeAggregateSumComponent<Config> &cpt;
  GraphInitialize ginit;
  GraphInitialize ginit_out;
  int graph_flat_size;
  int graph_out_flat_size;
  
  EdgeAggregateSumComponentWrapper(EdgeAggregateSumComponent<Config> &cpt, const GraphInitialize &ginit, const GraphInitialize &ginit_out, const int graph_flat_size, const int graph_out_flat_size): cpt(cpt), ginit(ginit), ginit_out(ginit_out), graph_flat_size(graph_flat_size), graph_out_flat_size(graph_out_flat_size){ }

  size_t outputLinearSize() const{ return graph_out_flat_size; }
  size_t inputLinearSize() const{ return graph_flat_size; }
  
  Vector<FloatType> value(const Vector<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    Graph<FloatType> ing(ginit);    
    unflatten(ing,in);
    return flatten( cpt.value(ing) );
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Graph<FloatType> above_deriv(ginit_out);
    unflatten(above_deriv, above_deriv_lin);   

    Graph<FloatType> cost_deriv_inputs_graph(ginit);
    cpt.deriv(std::move(above_deriv), cost_deriv_inputs_graph);
    cost_deriv_inputs = flatten(cost_deriv_inputs_graph);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i) const{
    return "";
  }      
};

void testEdgeAggregateSum(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  GraphInitialize ginit;
  ginit.nnode = 3;
  ginit.node_attr_sizes = std::vector<int>({3,4});
  ginit.edge_attr_sizes = std::vector<int>({5,6});
  //edges in circle
  ginit.edge_map = std::vector<std::pair<int,int> >({  {0,1}, {1,2}, {2,0}, {1,0}, {2,1}, {0,2} });
  ginit.global_attr_sizes = std::vector<int>({2});
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Tensor<FloatType,3> &m){ uniformRandom(m,rng); });

  EdgeAggregateSumComponent<Config> esum_cpt;
  Graph<FloatType> esum_got = esum_cpt.value(graph);
  Graph<FloatType> esum_expect = expectEdgeAggregateSum(graph);
  assert(equal(esum_got, esum_expect, true));
   
  EdgeAggregateSumComponentWrapper<Config> wrp(esum_cpt, graph.getInitializer(), esum_got.getInitializer(), flatSize(graph), flatSize(esum_got) );  
  testComponentDeriv(wrp, 1e-4, true);
  
  std::cout << "testEdgeAggregateSum passed" << std::endl;
}


template<typename FloatType>
Graph<FloatType> expectEdgeAggregateGlobalSum(const Graph<FloatType> &in){
  Graph<FloatType> out(in);
  for(int a=0;a<in.edges.nAttrib();a++)
    out.edges.attributes[a] = Tensor<FloatType,3>(1,in.edges.attribSize(a),in.edges.batchSize());

  out.edges.edge_map.resize(1);
  out.edges.edge_map[0].first = out.edges.edge_map[0].second = -1;

  autoView(out_edges_v,out.edges.attributes,HostWrite);
  autoView(in_edges_v,in.edges.attributes,HostRead);
  
  for(int a=0;a<in.edges.nAttrib();a++){
    for(int e=0;e<in.edges.nElem();e++){
      for(int i=0;i<in.edges.attribSize(a);i++)
	for(int b=0;b<in.edges.batchSize();b++)
	  if(e==0) out_edges_v[a](0,i,b) = in_edges_v[a](e,i,b);
	  else out_edges_v[a](0,i,b) += in_edges_v[a](e,i,b);
    }
  }

  return out;
}

template<typename Config>
struct EdgeAggregateGlobalSumComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  EdgeAggregateGlobalSumComponent<Config> &cpt;
  GraphInitialize ginit;
  GraphInitialize ginit_out;
  int graph_flat_size;
  int graph_out_flat_size;
  
  EdgeAggregateGlobalSumComponentWrapper(EdgeAggregateGlobalSumComponent<Config> &cpt, const GraphInitialize &ginit, const GraphInitialize &ginit_out, const int graph_flat_size, const int graph_out_flat_size): cpt(cpt), ginit(ginit), ginit_out(ginit_out), graph_flat_size(graph_flat_size), graph_out_flat_size(graph_out_flat_size){
  }

  size_t outputLinearSize() const{ return graph_out_flat_size; }
  size_t inputLinearSize() const{ return graph_flat_size; }
  
  Vector<FloatType> value(const Vector<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    Graph<FloatType> ing(ginit);    
    unflatten(ing,in);
    return flatten( cpt.value(ing) );
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Graph<FloatType> above_deriv(ginit_out);
    unflatten(above_deriv, above_deriv_lin);
    
    Graph<FloatType> cost_deriv_inputs_graph(ginit);
    cpt.deriv(std::move(above_deriv), cost_deriv_inputs_graph);
    cost_deriv_inputs = flatten(cost_deriv_inputs_graph);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i) const{
    return "";
  }      
};

void testEdgeAggregateGlobalSum(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  GraphInitialize ginit;
  ginit.nnode = 3;
  ginit.node_attr_sizes = std::vector<int>({3,4});
  ginit.edge_attr_sizes = std::vector<int>({5,6});
  //edges in circle
  ginit.edge_map = std::vector<std::pair<int,int> >({  {0,1}, {1,2}, {2,0}, {1,0}, {2,1}, {0,2} });
  ginit.global_attr_sizes = std::vector<int>({2});
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Tensor<FloatType,3> &m){ uniformRandom(m,rng); });

  EdgeAggregateGlobalSumComponent<Config> esum_cpt;
  Graph<FloatType> esum_got = esum_cpt.value(graph);
  Graph<FloatType> esum_expect = expectEdgeAggregateGlobalSum(graph);
  assert(equal(esum_got,esum_expect,true));
  
  EdgeAggregateGlobalSumComponentWrapper<Config> wrp(esum_cpt, graph.getInitializer(), esum_got.getInitializer(), flatSize(graph), flatSize(esum_got));  
  testComponentDeriv(wrp, 1e-4, true);
  
  std::cout << "testEdgeAggregateGlobalSum passed" << std::endl;
}
