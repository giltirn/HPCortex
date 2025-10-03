template<typename FloatType>
Tensor<FloatType,3> expectExtractEdgeUpdateInput(const Graph<FloatType> &in){
  int nedge = in.edges.nElem();
  int node_attr_total = 0;
  for(int a=0;a<in.nodes.nAttrib();a++)
    node_attr_total += in.nodes.attribSize(a);
  int edge_attr_total = 0;
  for(int a=0;a<in.edges.nAttrib();a++)
    edge_attr_total += in.edges.attribSize(a);
  int glob_attr_total = 0;
  for(int a=0;a<in.global.nAttrib();a++)
    glob_attr_total += in.global.attribSize(a);

  int batch_size = in.global.batchSize();

  int out_size[3] = { nedge, 2*node_attr_total + edge_attr_total + glob_attr_total, batch_size };

  Tensor<FloatType,3> out(out_size);
  autoView(out_v,out,HostWrite);
  autoView(edges_v,in.edges.attributes,HostRead);
  autoView(nodes_v,in.nodes.attributes,HostRead);
  autoView(global_v,in.global.attributes,HostRead);

  int nnode_attrib = in.nodes.nAttrib();
  int nedge_attrib = in.edges.nAttrib();
  int nglobal_attrib = in.global.nAttrib();
  
  for(int edge_idx=0; edge_idx < nedge; edge_idx++){
    int send_node = in.edges.sendNode(edge_idx);
    int recv_node = in.edges.recvNode(edge_idx);
    
    int dim1_off=0;
    for(int a=0;a<nedge_attrib;a++){
      int attrib_size = in.edges.attribSize(a);
      auto attr_v = edges_v[a];
      
      for(int i=0;i<attrib_size;i++)
	for(int b=0;b<batch_size;b++)
	  out_v(edge_idx,dim1_off +i,b) = attr_v(edge_idx,i,b);
      dim1_off += attrib_size;
    }
    for(int a=0;a<nnode_attrib;a++){
      int attrib_size = in.nodes.attribSize(a);
      auto attr_v = nodes_v[a];

      for(int i=0;i<attrib_size;i++)
	for(int b=0;b<batch_size;b++)
	  out_v(edge_idx,dim1_off +i,b) = attr_v(send_node,i,b);
      
      dim1_off += attrib_size;

      for(int i=0;i<attrib_size;i++)
	for(int b=0;b<batch_size;b++)
	  out_v(edge_idx,dim1_off +i,b) = attr_v(recv_node,i,b);

      dim1_off += attrib_size;
    }
    for(int a=0;a<nglobal_attrib;a++){    
      int attrib_size = in.global.attribSize(a);
      auto attr_v = global_v[a];
      
      for(int i=0;i<attrib_size;i++)
	for(int b=0;b<batch_size;b++)
	  out_v(edge_idx,dim1_off +i,b) = attr_v(0,i,b);
      dim1_off += attrib_size;
    }
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
  int tens_sz[2];
  
  ExtractEdgeUpdateInputComponentWrapper(ExtractEdgeUpdateInputComponent<Config> &cpt, const GraphInitialize &ginit, const int graph_flat_size, const int _tens_sz[2]): cpt(cpt), ginit(ginit), graph_flat_size(graph_flat_size){
    memcpy(tens_sz, _tens_sz, 2*sizeof(int));
    tens_flat_size = tens_sz[0]*tens_sz[1];
  }

  size_t outputLinearSize() const{ return tens_flat_size; }
  size_t inputLinearSize() const{ return graph_flat_size; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    ginit.batch_size = in.size(1);
    Graph<FloatType> graph = unflattenFromBatchVector(in, ginit);
    return flattenToBatchVector(cpt.value(graph));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    batchTensorSize(tens_sz_b, 3, tens_sz, _above_deriv_lin.size(1));
    Tensor<FloatType,3> above_deriv = unflattenFromBatchVector<3>(_above_deriv_lin, tens_sz_b);

    ginit.batch_size = _above_deriv_lin.size(1);
    Graph<FloatType> cost_deriv_inputs_graph(ginit);
    cpt.deriv(std::move(above_deriv), cost_deriv_inputs_graph);
    cost_deriv_inputs = flattenToBatchVector(cost_deriv_inputs_graph);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i, int b, int batch_size) const{
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
  ginit.global_attr_sizes = std::vector<int>({2});
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Tensor<FloatType,3> &m){ uniformRandom(m,rng); });

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
  
  ExtractEdgeUpdateInputComponentWrapper<Config> wrp(eup_cpt, graph.getInitializer(), rowsAsBatchVector(graph), eup_in.sizeArray());
  testComponentDeriv(wrp, ginit.batch_size, 1e-4, true);
  testComponentDiffBatchSizes(wrp);
  
  std::cout << "testExtractEdgeUpdateInputComponent passed" << std::endl;
}


template<typename FloatType>
Graph<FloatType> expectInsertEdgeUpdateOutput(const Graph<FloatType> &in, const Tensor<FloatType,3> &edge_attr_update){
  int nedge = in.edges.nElem();
  int nedge_attr = in.edges.nAttrib();
  int edge_attr_total = 0;
  for(int a=0;a<nedge_attr;a++)
    edge_attr_total += in.edges.attribSize(a);
  int batch_size = in.nodes.batchSize();

  Graph<FloatType> out(in);
  autoView(tin_v,edge_attr_update,HostRead);
  autoView(edges_v,out.edges.attributes,HostWrite);

  int dim1_off = 0;
  for(int a=0;a<nedge_attr;a++){
    int attr_sz = in.edges.attribSize(a);      
    auto attr_v = edges_v[a];
    for(int edge_idx=0; edge_idx < nedge; edge_idx++)
      for(int i=0;i<attr_sz;i++)
	for(int b=0;b<batch_size;b++)
	  attr_v(edge_idx,i,b) = tin_v(edge_idx,dim1_off +i,b);
    dim1_off += attr_sz;
  }

  return out;
}


template<typename Config>
struct InsertEdgeUpdateOutputComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  InsertEdgeUpdateOutputComponent<Config> &cpt;
  GraphInitialize ginit;
  int graph_flat_size;
  int tens_flat_size;
  int tens_sz[2];
  
  InsertEdgeUpdateOutputComponentWrapper(InsertEdgeUpdateOutputComponent<Config> &cpt, const GraphInitialize &ginit, const int graph_flat_size, const int _tens_sz[2]): cpt(cpt), ginit(ginit), graph_flat_size(graph_flat_size){
    memcpy(tens_sz, _tens_sz, 2*sizeof(int));
    tens_flat_size = tens_sz[0]*tens_sz[1];
  }

  size_t outputLinearSize() const{ return graph_flat_size; }
  size_t inputLinearSize() const{ return graph_flat_size + tens_flat_size; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    ginit.batch_size = in.size(1);
    Graph<FloatType> graph(ginit);
    
    int poff = unflattenFromBatchVector(graph, in, 0);
    batchTensorSize(tens_sz_b, 3, tens_sz, in.size(1));
    
    Tensor<FloatType,3> edge_attr_update(tens_sz_b);
    unflattenFromBatchVector(edge_attr_update, in, poff);
    return flattenToBatchVector(cpt.value(graph, edge_attr_update));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size=  _above_deriv_lin.size(1);
    ginit.batch_size = batch_size;
    batchTensorSize(tens_sz_b, 3, tens_sz, batch_size);
    
    Graph<FloatType> above_deriv = unflattenFromBatchVector(_above_deriv_lin, ginit);
    
    Tensor<FloatType,3> cost_deriv_inputs_tens(tens_sz_b);
    Graph<FloatType> cost_deriv_inputs_graph(ginit);
    cpt.deriv(std::move(above_deriv), cost_deriv_inputs_graph, cost_deriv_inputs_tens);

    cost_deriv_inputs = Matrix<FloatType>(inputLinearSize(), batch_size);
    int poff = flattenToBatchVector(cost_deriv_inputs, cost_deriv_inputs_graph, 0);
    flattenToBatchVector(cost_deriv_inputs, cost_deriv_inputs_tens, poff);
  }
    
  void update(int off, const Vector<FloatType> &new_params){}
  void step(int off, const Vector<FloatType> &derivs, FloatType eps){}
  inline int nparams() const{ return cpt.nparams(); }
  void getParams(Vector<FloatType> &into, int off){}

  std::string inCoord(size_t i, int b, int batch_size) const{
    return "";
  }      
};

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
  ginit.global_attr_sizes = std::vector<int>({2});
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Tensor<FloatType,3> &m){ uniformRandom(m,rng); });

  InsertEdgeUpdateOutputComponent<Config> eup_cpt;

  int edge_attr_total_size = 5+6;

  Tensor<FloatType,3> eup_in(ginit.edge_map.size(), edge_attr_total_size, ginit.batch_size);
  uniformRandom(eup_in, rng);

  Graph<FloatType> gup = eup_cpt.value(graph, eup_in);
  Graph<FloatType> gexpect = expectInsertEdgeUpdateOutput(graph, eup_in);
  assert(equal(gup, gexpect,true));

  InsertEdgeUpdateOutputComponentWrapper<Config> wrp(eup_cpt, ginit, rowsAsBatchVector(graph), eup_in.sizeArray());
  testComponentDeriv(wrp, ginit.batch_size, 1e-4, true);
  testComponentDiffBatchSizes(wrp);
  
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
  ginit.global_attr_sizes = std::vector<int>({2});
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Tensor<FloatType,3> &m){ uniformRandom(m,rng); });

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
  
  GraphInGraphOutLayerWrapper<Config, decltype(eup_out)> wrp(eup_out, ginit, rowsAsBatchVector(graph));
  testComponentDeriv(wrp, ginit.batch_size, 1e-4, true);
  testComponentDiffBatchSizes(wrp);
  
  std::cout << "testEdgeUpdateBlock passed" << std::endl;
}
