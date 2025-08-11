template<typename FloatType>
Tensor<FloatType,3> expectExtractNodeUpdateInput(const Graph<FloatType> &in){
  int nnode = in.nodes.size();
  int node_attr_total = 0;
  for(int a=0;a<in.nodes[0].attributes.size();a++)
    node_attr_total += in.nodes[0].attributes[a].size(0);
  int edge_attr_total = 0;
  for(int a=0;a<in.edges[0].attributes.size();a++)
    edge_attr_total += in.edges[0].attributes[a].size(0);
  int glob_attr_total = in.global.size(0);
  int batch_size = in.global.size(1);

  int out_size[3] = { nnode, node_attr_total + edge_attr_total + glob_attr_total, batch_size };

  Tensor<FloatType,3> out(out_size);
  autoView(out_v,out,HostWrite);
  for(int node_idx=0; node_idx < nnode; node_idx++){
    const Edge<FloatType> &agg_edge = in.edges[ node_idx ];
    const Node<FloatType> &node = in.nodes[ node_idx ];

    int dim1_off=0;
    for(int a=0;a<node.attributes.size();a++){
      autoView(attr_v, node.attributes[a], HostRead);
      for(int i=0;i<attr_v.size(0);i++)
	for(int b=0;b<batch_size;b++)
	  out_v(node_idx,dim1_off +i,b) = attr_v(i,b);
      dim1_off += attr_v.size(0);
    }
    for(int a=0;a<agg_edge.attributes.size();a++){
      autoView(attr_v, agg_edge.attributes[a], HostRead);
      for(int i=0;i<attr_v.size(0);i++)
	for(int b=0;b<batch_size;b++)
	  out_v(node_idx,dim1_off +i,b) = attr_v(i,b);
      dim1_off += attr_v.size(0);
    }
    autoView(attr_v, in.global, HostRead);
    for(int i=0;i<attr_v.size(0);i++)
      for(int b=0;b<batch_size;b++)
	out_v(node_idx,dim1_off +i,b) = attr_v(i,b);
    dim1_off += attr_v.size(0);
    assert(dim1_off == out_size[1]);
  }
  return out;
}

template<typename Config>
struct ExtractNodeUpdateInputComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  ExtractNodeUpdateInputComponent<Config> &cpt;
  GraphInitialize ginit;
  int graph_flat_size;
  int tens_flat_size;
  int tens_sz[3];
  
  ExtractNodeUpdateInputComponentWrapper(ExtractNodeUpdateInputComponent<Config> &cpt, const GraphInitialize &ginit, const int graph_flat_size, const int _tens_sz[3]): cpt(cpt), ginit(ginit), graph_flat_size(graph_flat_size){
    memcpy(tens_sz, _tens_sz, 3*sizeof(int));
    tens_flat_size = 1;
    for(int i=0;i<3;i++)
      tens_flat_size *= tens_sz[i];
  }

  size_t outputLinearSize() const{ return tens_flat_size; }
  size_t inputLinearSize() const{ return graph_flat_size; }
  
  Vector<FloatType> value(const Vector<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    Graph<FloatType> graph(ginit);
    unflatten(graph, in);
    Tensor<FloatType,3> out = cpt.value(graph);
    return flatten(out);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Tensor<FloatType,3> above_deriv(tens_sz);
    unflatten(above_deriv,above_deriv_lin);
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


void testExtractNodeUpdateInputComponent(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  GraphInitialize ginit;
  ginit.nnode = 3;
  ginit.node_attr_sizes = std::vector<int>({3,4});
  ginit.edge_attr_sizes = std::vector<int>({5});

  //here the edges are aggregated edges, one per node
  ginit.edge_map = std::vector<std::pair<int,int> >({  {-1,0}, {-1,1}, {-1,2} });
  ginit.global_attr_size = 2;
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Matrix<FloatType> &m){ uniformRandom(m,rng); });

  ExtractNodeUpdateInputComponent<Config> eup_cpt;
  Tensor<FloatType,3> eup_in = eup_cpt.value(graph);
  Tensor<FloatType,3> eup_in_expect = expectExtractNodeUpdateInput(graph);
  assert(equal(eup_in_expect, eup_in, true)); 
  
  ExtractNodeUpdateInputComponentWrapper<Config> wrp(eup_cpt, graph.getInitializer(), flatSize(graph), eup_in.sizeArray());
  testComponentDeriv(wrp, 1e-4, true);
  
  std::cout << "testExtractNodeUpdateInputComponent passed" << std::endl;
}

template<typename Config>
struct InsertNodeUpdateOutputComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  InsertNodeUpdateOutputComponent<Config> &cpt;
  GraphInitialize ginit;
  int graph_flat_size;
  int tens_flat_size;
  int tens_sz[3];
  
  InsertNodeUpdateOutputComponentWrapper(InsertNodeUpdateOutputComponent<Config> &cpt, const GraphInitialize &ginit, const int graph_flat_size, const int _tens_sz[3]): cpt(cpt), ginit(ginit), graph_flat_size(graph_flat_size){
    memcpy(tens_sz, _tens_sz, 3*sizeof(int));
    tens_flat_size = 1;
    for(int i=0;i<3;i++)
      tens_flat_size *= tens_sz[i];
  }

  size_t outputLinearSize() const{ return graph_flat_size; }
  size_t inputLinearSize() const{ return graph_flat_size + tens_flat_size; }
  
  Vector<FloatType> value(const Vector<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    autoView(in_v,in,HostRead);
    FloatType const* p = in_v.data();    
    Graph<FloatType> graph(ginit);
    p = unflatten(graph, p);
    
    Tensor<FloatType,3> node_attr_update(tens_sz);
    p = unflatten(node_attr_update, p);

    Graph<FloatType> out = cpt.value(graph, node_attr_update);
    return flatten(out);
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Vector<FloatType> &&_above_deriv_lin, Vector<FloatType> &cost_deriv_inputs){
    Vector<FloatType> above_deriv_lin = std::move(_above_deriv_lin);
    Graph<FloatType> above_deriv(ginit);
    unflatten(above_deriv, above_deriv_lin);
    Tensor<FloatType,3> cost_deriv_inputs_tens(tens_sz);
    Graph<FloatType> cost_deriv_inputs_graph(ginit);
    cpt.deriv(std::move(above_deriv), cost_deriv_inputs_graph, cost_deriv_inputs_tens);

    cost_deriv_inputs = Vector<FloatType>(inputLinearSize());
    autoView(out_v, cost_deriv_inputs, HostWrite);
    FloatType *p = out_v.data();
    p = flatten(p,cost_deriv_inputs_graph);
    p = flatten(p,cost_deriv_inputs_tens);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i) const{
    return "";
  }      
};





template<typename FloatType>
Graph<FloatType> expectInsertNodeUpdateOutput(const Graph<FloatType> &in, const Tensor<FloatType,3> &node_attr_update){
  int nnode = in.nodes.size();
  int node_attr_total = 0;
  for(int a=0;a<in.nodes[0].attributes.size();a++)
    node_attr_total += in.nodes[0].attributes[a].size(0);
  int batch_size = in.global.size(1);

  Graph<FloatType> out(in);
  autoView(tin_v,node_attr_update,HostRead);
  for(int node_idx=0; node_idx < nnode; node_idx++){
    Node<FloatType> &node = out.nodes[node_idx];
    
    int dim1_off = 0;
    for(int a=0;a<node.attributes.size();a++){
      autoView(attr_v, node.attributes[a], HostWrite);
      for(int i=0;i<attr_v.size(0);i++)
	for(int b=0;b<batch_size;b++)
	  attr_v(i,b) = tin_v(node_idx,dim1_off +i,b);
      dim1_off += attr_v.size(0);
    }
  }
  return out;
}

void testInsertNodeUpdateOutput(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  GraphInitialize ginit;
  ginit.nnode = 3;
  ginit.node_attr_sizes = std::vector<int>({3,4});
  ginit.edge_attr_sizes = std::vector<int>({5,6});
  //here the edges are aggregated edges, one per node
  ginit.edge_map = std::vector<std::pair<int,int> >({  {-1,0}, {-1,1}, {-1,2} });
  ginit.global_attr_size = 2;
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Matrix<FloatType> &m){ uniformRandom(m,rng); });

  InsertNodeUpdateOutputComponent<Config> eup_cpt;

  int node_attr_total_size = 3+4;

  Tensor<FloatType,3> eup_in(ginit.nnode, node_attr_total_size, ginit.batch_size);
  uniformRandom(eup_in, rng);

  Graph<FloatType> gup = eup_cpt.value(graph, eup_in);
  Graph<FloatType> gexpect = expectInsertNodeUpdateOutput(graph, eup_in);
  assert(equal(gup, gexpect,true));

  InsertNodeUpdateOutputComponentWrapper<Config> wrp(eup_cpt, ginit, flatSize(graph), eup_in.sizeArray());
  testComponentDeriv(wrp, 1e-4, true);
  
  std::cout << "testInsertNodeUpdateOutput passed" << std::endl;
}

void testNodeUpdateBlock(){
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

  typedef Graph<FloatType> InputType;
   
  auto splt = replicate_layer(2, input_layer<Config, InputType>());
  auto agg = edge_aggregate_sum_layer(*splt[0]);
  auto nup_in = extract_node_update_input_layer(agg);
  std::array<int,3> nup_in_tens_sz = ExtractNodeUpdateInputComponent<Config>::outputTensorSize(ginit);
  std::array<int,3> nup_out_tens_sz = InsertNodeUpdateOutputComponent<Config>::inputTensorSize(ginit);

  Matrix<FloatType> weight(nup_out_tens_sz[1], nup_in_tens_sz[1]);
  Vector<FloatType> bias(nup_out_tens_sz[1]);
  uniformRandom(weight);
  uniformRandom(bias);
  
  auto nup = batch_tensor_dnn_layer<3>(weight, bias, 1, noActivation<FloatType>(), nup_in );  
  auto nup_out = insert_node_update_output_layer(*splt[1], nup);

  //compute expectation
  Graph<FloatType> expect;
  {
    Graph<FloatType> gin = expectEdgeAggregateSum(graph);
    Tensor<FloatType,3> tens_in = expectExtractNodeUpdateInput(gin);
    Tensor<FloatType,3> tens_out = matrixBatchTensorAxpy(weight, tens_in, bias, 1);
    expect = expectInsertNodeUpdateOutput(graph, tens_out);
  }
  Graph<FloatType> got = nup_out.value(graph, DerivNo);
  assert(abs_near(got,expect,FloatType(1e-5),true));
  
  GraphInGraphOutLayerWrapper<Config, decltype(nup_out)> wrp(nup_out, ginit, flatSize(graph));
  testComponentDeriv(wrp, 1e-4, true);
      
  std::cout << "testNodeUpdateBlock passed" << std::endl;
}
