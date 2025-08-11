


template<typename FloatType>
Graph<FloatType> expectEdgeAggregateSum(const Graph<FloatType> &in){
  Graph<FloatType> out(in);
  out.edges.resize(in.nodes.size());

  for(int n=0;n<in.nodes.size();n++){
    Edge<FloatType> &edge_out = out.edges[n];
    edge_out.send_node = -1;
    edge_out.recv_node = n;
    edge_out.attributes.resize( in.edges[0].attributes.size() );

    bool first = true;
    for(auto const &edge_in : in.edges)
      if(edge_in.recv_node == n){
	for(int a=0;a<edge_in.attributes.size();a++){
	  if(first) edge_out.attributes[a] = edge_in.attributes[a];
	  else edge_out.attributes[a] += edge_in.attributes[a];
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
  ginit.global_attr_size = 2;
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Matrix<FloatType> &m){ uniformRandom(m,rng); });

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
  out.edges.resize(1);
  
  Edge<FloatType> &edge_out = out.edges[0];
  edge_out.send_node = -1;
  edge_out.recv_node = -1;
  edge_out.attributes.resize( in.edges[0].attributes.size() );

  bool first = true;
  for(int e=0;e<in.edges.size();e++){
    for(int a=0;a<in.edges[e].attributes.size();a++){
      if(first) edge_out.attributes[a] = in.edges[e].attributes[a];
      else edge_out.attributes[a] += in.edges[e].attributes[a];
    }
    first = false;
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
  ginit.global_attr_size = 2;
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Matrix<FloatType> &m){ uniformRandom(m,rng); });

  EdgeAggregateGlobalSumComponent<Config> esum_cpt;
  Graph<FloatType> esum_got = esum_cpt.value(graph);
  Graph<FloatType> esum_expect = expectEdgeAggregateGlobalSum(graph);
  assert(equal(esum_got,esum_expect,true));
  
  EdgeAggregateGlobalSumComponentWrapper<Config> wrp(esum_cpt, graph.getInitializer(), esum_got.getInitializer(), flatSize(graph), flatSize(esum_got));  
  testComponentDeriv(wrp, 1e-4, true);
  
  std::cout << "testEdgeAggregateGlobalSum passed" << std::endl;
}
