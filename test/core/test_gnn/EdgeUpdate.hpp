


template<typename FloatType>
Tensor<FloatType,3> expectExtractEdgeUpdateInput(const Graph<FloatType> &in){
  int nedge = in.edges.size();
  int node_attr_total = 0;
  for(int a=0;a<in.nodes[0].attributes.size();a++)
    node_attr_total += in.nodes[0].attributes[a].size(0);
  int edge_attr_total = 0;
  for(int a=0;a<in.edges[0].attributes.size();a++)
    edge_attr_total += in.edges[0].attributes[a].size(0);
  int glob_attr_total = in.global.size(0);
  int batch_size = in.global.size(1);

  int out_size[3] = { nedge, 2*node_attr_total + edge_attr_total + glob_attr_total, batch_size };

  Tensor<FloatType,3> out(out_size);
  autoView(out_v,out,HostWrite);
  for(int edge_idx=0; edge_idx < nedge; edge_idx++){
    const Edge<FloatType> &edge = in.edges[edge_idx];
    const Node<FloatType> &send_node = in.nodes[ edge.send_node ];
    const Node<FloatType> &recv_node = in.nodes[ edge.recv_node ];

    int dim1_off=0;
    for(int a=0;a<send_node.attributes.size();a++){
      autoView(attr_v, send_node.attributes[a], HostRead);
      for(int i=0;i<attr_v.size(0);i++)
	for(int b=0;b<batch_size;b++)
	  out_v(edge_idx,dim1_off +i,b) = attr_v(i,b);
      dim1_off += attr_v.size(0);
    }
    for(int a=0;a<recv_node.attributes.size();a++){
      autoView(attr_v, recv_node.attributes[a], HostRead);
      for(int i=0;i<attr_v.size(0);i++)
	for(int b=0;b<batch_size;b++)
	  out_v(edge_idx,dim1_off +i,b) = attr_v(i,b);
      dim1_off += attr_v.size(0);
    }
    for(int a=0;a<edge.attributes.size();a++){
      autoView(attr_v, edge.attributes[a], HostRead);
      for(int i=0;i<attr_v.size(0);i++)
	for(int b=0;b<batch_size;b++)
	  out_v(edge_idx,dim1_off +i,b) = attr_v(i,b);
      dim1_off += attr_v.size(0);
    }
    autoView(attr_v, in.global, HostRead);
    for(int i=0;i<attr_v.size(0);i++)
      for(int b=0;b<batch_size;b++)
	out_v(edge_idx,dim1_off +i,b) = attr_v(i,b);
    dim1_off += attr_v.size(0);
    assert(dim1_off == out_size[1]);
  }
  return out;
}

template<typename Config>
struct ExtractEdgeUpdateInputComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  ExtractEdgeUpdateInputComponent<Config> &cpt;
  GraphInitialize ginit;
  int graph_flat_size;
  int tens_flat_size;
  int tens_sz[3];
  
  ExtractEdgeUpdateInputComponentWrapper(ExtractEdgeUpdateInputComponent<Config> &cpt, const GraphInitialize &ginit, const int graph_flat_size, const int _tens_sz[3]): cpt(cpt), ginit(ginit), graph_flat_size(graph_flat_size){
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


void testExtractEdgeUpdateInputComponent(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  GraphInitialize ginit;
  ginit.nnode = 3;
  ginit.node_attr_sizes = std::vector<int>({3,4});
  ginit.edge_attr_sizes = std::vector<int>({5});
  //edges in circle
  ginit.edge_map = std::vector<std::pair<int,int> >({  {0,1}, {1,2}, {2,0}, {1,0}, {2,1}, {0,2} });
  ginit.global_attr_size = 2;
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Matrix<FloatType> &m){ uniformRandom(m,rng); });

  ExtractEdgeUpdateInputComponent<Config> eup_cpt;
  Tensor<FloatType,3> eup_in = eup_cpt.value(graph);
  Tensor<FloatType,3> eup_in_expect = expectExtractEdgeUpdateInput(graph);
  assert(equal(eup_in_expect, eup_in, true));
  
  //test tests!
  assert(equal(graph, graph,false));
  assert(abs_near(graph, graph, FloatType(1e-8), false));

  Graph<FloatType> test_graph;
  assert(!equal(test_graph, graph,false));
  assert(!abs_near(test_graph, graph, FloatType(1e-8), false));
  test_graph = Graph<FloatType>(graph.getInitializer());
  assert(!equal(test_graph, graph,false));
  assert(!abs_near(test_graph, graph, FloatType(1e-8), false));
  test_graph = graph;
  assert(equal(test_graph, graph,false));
  assert(abs_near(test_graph, graph, FloatType(1e-8), false));

  //test flatten/unflatten
  Vector<FloatType> gflat = flatten(graph);
  Graph<FloatType> gunflat(graph.getInitializer());
  unflatten(gunflat,gflat);
  assert(equal(graph,gunflat,true));
  
  
  ExtractEdgeUpdateInputComponentWrapper<Config> wrp(eup_cpt, graph.getInitializer(), flatSize(graph), eup_in.sizeArray());
  testComponentDeriv(wrp, 1e-4, true);
  
  std::cout << "testExtractEdgeUpdateInputComponent passed" << std::endl;
}






template<typename Config>
struct InsertEdgeUpdateOutputComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  InsertEdgeUpdateOutputComponent<Config> &cpt;
  GraphInitialize ginit;
  int graph_flat_size;
  int tens_flat_size;
  int tens_sz[3];
  
  InsertEdgeUpdateOutputComponentWrapper(InsertEdgeUpdateOutputComponent<Config> &cpt, const GraphInitialize &ginit, const int graph_flat_size, const int _tens_sz[3]): cpt(cpt), ginit(ginit), graph_flat_size(graph_flat_size){
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
    
    Tensor<FloatType,3> edge_attr_update(tens_sz);
    p = unflatten(edge_attr_update, p);
    Graph<FloatType> out = cpt.value(graph, edge_attr_update);
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
Graph<FloatType> expectInsertEdgeUpdateOutput(const Graph<FloatType> &in, const Tensor<FloatType,3> &edge_attr_update){
  int nedge = in.edges.size();
  int edge_attr_total = 0;
  for(int a=0;a<in.edges[0].attributes.size();a++)
    edge_attr_total += in.edges[0].attributes[a].size(0);
  int batch_size = in.global.size(1);

  Graph<FloatType> out(in);
  autoView(tin_v,edge_attr_update,HostRead);
  for(int edge_idx=0; edge_idx < nedge; edge_idx++){
    Edge<FloatType> &edge = out.edges[edge_idx];
    
    int dim1_off = 0;
    for(int a=0;a<edge.attributes.size();a++){
      autoView(attr_v, edge.attributes[a], HostWrite);
      for(int i=0;i<attr_v.size(0);i++)
	for(int b=0;b<batch_size;b++)
	  attr_v(i,b) = tin_v(edge_idx,dim1_off +i,b);
      dim1_off += attr_v.size(0);
    }
  }
  return out;
}

void testInsertEdgeUpdateOutput(){
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

  InsertEdgeUpdateOutputComponent<Config> eup_cpt;

  int edge_attr_total_size = 5+6;

  Tensor<FloatType,3> eup_in(ginit.edge_map.size(), edge_attr_total_size, ginit.batch_size);
  uniformRandom(eup_in, rng);

  Graph<FloatType> gup = eup_cpt.value(graph, eup_in);
  Graph<FloatType> gexpect = expectInsertEdgeUpdateOutput(graph, eup_in);
  assert(equal(gup, gexpect,true));

  InsertEdgeUpdateOutputComponentWrapper<Config> wrp(eup_cpt, ginit, flatSize(graph), eup_in.sizeArray());
  testComponentDeriv(wrp, 1e-4, true);
  
  std::cout << "testInsertEdgeUpdateOutput passed" << std::endl;
}

void testEdgeUpdateBlock(){
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
  auto eup_in = extract_edge_update_input_layer(*splt[0]);
  std::array<int,3> eup_in_tens_sz = ExtractEdgeUpdateInputComponent<Config>::outputTensorSize(ginit);
  std::array<int,3> eup_out_tens_sz = InsertEdgeUpdateOutputComponent<Config>::inputTensorSize(ginit);

  Matrix<FloatType> weight(eup_out_tens_sz[1], eup_in_tens_sz[1]);
  Vector<FloatType> bias(eup_out_tens_sz[1]);
  uniformRandom(weight);
  uniformRandom(bias);
  
  auto eup = batch_tensor_dnn_layer<3>(weight, bias, 1, noActivation<FloatType>(), eup_in );  
  auto eup_out = insert_edge_update_output_layer(*splt[1], eup);

  //compute expectation
  Graph<FloatType> expect;
  {
    Tensor<FloatType,3> tens_in = expectExtractEdgeUpdateInput(graph);
    Tensor<FloatType,3> tens_out = matrixBatchTensorAxpy(weight, tens_in, bias, 1);
    expect = expectInsertEdgeUpdateOutput(graph, tens_out);
  }
  Graph<FloatType> got = eup_out.value(graph, DerivNo);
  assert(abs_near(got,expect,FloatType(1e-5),true));
  
  GraphInGraphOutLayerWrapper<Config, decltype(eup_out)> wrp(eup_out, ginit, flatSize(graph));
  testComponentDeriv(wrp, 1e-4, true);
      
  std::cout << "testEdgeUpdateBlock passed" << std::endl;
}
