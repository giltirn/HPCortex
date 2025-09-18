template<typename FloatType>
Graph<FloatType> expectNodeAggregateGlobalSum(const Graph<FloatType> &in){
  Graph<FloatType> out(in);
  for(int a=0;a<in.nodes.nAttrib();a++)
    out.nodes.attributes[a] = Tensor<FloatType,3>(1,in.nodes.attribSize(a),in.nodes.batchSize());

  autoView(out_attr_v, out.nodes.attributes, HostWrite);
  autoView(in_attr_v, in.nodes.attributes, HostRead);
  
  for(int a=0;a<in.nodes.nAttrib();a++){
    for(int n=0;n<in.nodes.nElem();n++)
      for(int i=0;i<in.nodes.attribSize(a);i++)
	for(int b=0;b<in.nodes.batchSize();b++)
	  if(n==0) out_attr_v[a](0,i,b) = in_attr_v[a](n,i,b);
	  else out_attr_v[a](0,i,b) += in_attr_v[a](n,i,b);
  }
  return out;
}

template<typename Config>
struct NodeAggregateGlobalSumComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  NodeAggregateGlobalSumComponent<Config> &cpt;
  GraphInitialize ginit;
  GraphInitialize ginit_out;  
  int graph_flat_size;
  int graph_flat_size_out;
     
  NodeAggregateGlobalSumComponentWrapper(NodeAggregateGlobalSumComponent<Config> &cpt, const GraphInitialize &ginit, const GraphInitialize &ginit_out, const int graph_flat_size, const int graph_flat_size_out): cpt(cpt), ginit(ginit), ginit_out(ginit_out), graph_flat_size(graph_flat_size), graph_flat_size_out(graph_flat_size_out){
  }

  size_t outputLinearSize() const{ return graph_flat_size_out; }
  size_t inputLinearSize() const{ return graph_flat_size; }
  
  Vector<FloatType> value(const Vector<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    Graph<FloatType> ing(ginit);    
    unflatten(ing,in);
    return flatten(cpt.value(ing));
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

void testNodeAggregateGlobalSum(){
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

  NodeAggregateGlobalSumComponent<Config> esum_cpt;
  Graph<FloatType> esum_got = esum_cpt.value(graph);
  Graph<FloatType> esum_expect = expectNodeAggregateGlobalSum(graph);
  assert(equal(esum_got,esum_expect,true));
  
  NodeAggregateGlobalSumComponentWrapper<Config> wrp(esum_cpt, graph.getInitializer(), esum_got.getInitializer(), flatSize(graph), flatSize(esum_got));  
  testComponentDeriv(wrp, 1e-4, true);
  
  std::cout << "testNodeAggregateGlobalSum passed" << std::endl;
}

