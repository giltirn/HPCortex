template<typename FloatType>
Tensor<FloatType,2> expectExtractGlobalUpdateInput(const Graph<FloatType> &in){
  assert(in.nodes.nElem() == 1 && in.edges.nElem() == 1);

  int node_attr_total = 0;
  for(int a=0;a<in.nodes.nAttrib();a++)
    node_attr_total += in.nodes.attribSize(a);
  int edge_attr_total = 0;
  for(int a=0;a<in.edges.nAttrib();a++)
    edge_attr_total += in.edges.attribSize(a);
  int glob_attr_total = 0;
  for(int a=0;a<in.global.nAttrib();a++)
    glob_attr_total += in.global.attribSize(a);

  int batch_size = in.nodes.batchSize();

  int out_size[2] = { node_attr_total + edge_attr_total + glob_attr_total, batch_size };

  Tensor<FloatType,2> out(out_size);
  autoView(out_v,out,HostWrite);

  autoView(edges_v,in.edges.attributes,HostRead);
  autoView(nodes_v,in.nodes.attributes,HostRead);
  autoView(global_v,in.global.attributes,HostRead);
  
  int dim0_off=0;
  for(int a=0;a<in.nodes.nAttrib();a++){
    auto attr_v = nodes_v[a];
    int attr_sz = in.nodes.attribSize(a);
    for(int i=0;i<attr_sz;i++)
      for(int b=0;b<batch_size;b++)
	out_v(dim0_off +i,b) = attr_v(0,i,b);
    dim0_off += attr_sz;
  }
  for(int a=0;a<in.edges.nAttrib();a++){
    auto attr_v = edges_v[a];
    int attr_sz = in.edges.attribSize(a);
    for(int i=0;i<attr_sz;i++)
      for(int b=0;b<batch_size;b++)
	out_v(dim0_off +i,b) = attr_v(0,i,b);
    dim0_off += attr_sz;
  }
  for(int a=0;a<in.global.nAttrib();a++){
    auto attr_v = global_v[a];
    int attr_sz = in.global.attribSize(a);
    for(int i=0;i<attr_sz;i++)
      for(int b=0;b<batch_size;b++)
	out_v(dim0_off +i,b) = attr_v(0,i,b);
    dim0_off += attr_sz;
  }
  assert(dim0_off == out_size[0]);

  return out;
}

template<typename Config>
struct ExtractGlobalUpdateInputComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  ExtractGlobalUpdateInputComponent<Config> &cpt;
  GraphInitialize ginit;
  int graph_flat_size;
  int tens_sz;
  
  ExtractGlobalUpdateInputComponentWrapper(ExtractGlobalUpdateInputComponent<Config> &cpt, const GraphInitialize &ginit, const int graph_flat_size, const int _tens_sz): cpt(cpt), ginit(ginit), graph_flat_size(graph_flat_size), tens_sz(_tens_sz){
  }

  size_t outputLinearSize() const{ return tens_sz; }
  size_t inputLinearSize() const{ return graph_flat_size; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    ginit.batch_size = in.size(1);
    Graph<FloatType> graph = unflattenFromBatchVector(in, ginit);
    return flattenToBatchVector(cpt.value(graph));
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size = _above_deriv_lin.size(1);
    Tensor<FloatType,2> above_deriv(tens_sz, batch_size);
    unflattenFromBatchVector(above_deriv,_above_deriv_lin,0);
    Graph<FloatType> cost_deriv_inputs_graph;
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

void testExtractGlobalUpdateInputComponent(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  GraphInitialize ginit;
  //here the nodes are aggregated
  ginit.nnode = 1;
  ginit.node_attr_sizes = std::vector<int>({3,4});
  ginit.edge_attr_sizes = std::vector<int>({5});

  //here the edges are aggregated 
  ginit.edge_map = std::vector<std::pair<int,int> >({  {-1,-1} });
  ginit.global_attr_sizes = std::vector<int>({2});
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Tensor<FloatType,3> &m){ uniformRandom(m,rng); });

  ExtractGlobalUpdateInputComponent<Config> eup_cpt;
  Tensor<FloatType,2> eup_in = eup_cpt.value(graph);
  Tensor<FloatType,2> eup_in_expect = expectExtractGlobalUpdateInput(graph);
  assert(equal(eup_in_expect, eup_in, true)); 
  
  ExtractGlobalUpdateInputComponentWrapper<Config> wrp(eup_cpt, graph.getInitializer(), rowsAsBatchVector(graph), eup_in.size(0));
  testComponentDeriv(wrp, ginit.batch_size, 1e-4, true);
  testComponentDiffBatchSizes(wrp);
  std::cout << "testExtractGlobalUpdateInputComponent passed" << std::endl;
}

template<typename FloatType>
Graph<FloatType> expectInsertGlobalUpdateOutput(const Graph<FloatType> &in, const Tensor<FloatType,2> &glob_attr_update){
  Graph<FloatType> out(in);
  
  autoView(tin_v,glob_attr_update,HostRead);
  autoView(glob_attr_v, out.global.attributes, HostWrite);
  int batch_size=out.nodes.batchSize();
  
  int dim1_off = 0;
  for(int a=0;a<out.global.nAttrib();a++){
    auto attr_v = glob_attr_v[a];
    int attr_sz = out.global.attribSize(a);

    for(int i=0;i<attr_sz;i++)
      for(int b=0;b<batch_size;b++)
	attr_v(0,i,b) = tin_v(dim1_off +i,b);
    dim1_off += attr_sz;
  }

  return out;
}


template<typename Config>
struct InsertGlobalUpdateOutputComponentWrapper{
  EXTRACT_CONFIG_TYPES;
  
  InsertGlobalUpdateOutputComponent<Config> &cpt;
  GraphInitialize ginit;
  int graph_flat_size;
  int tens_sz;
  
  InsertGlobalUpdateOutputComponentWrapper(InsertGlobalUpdateOutputComponent<Config> &cpt, const GraphInitialize &ginit, const int graph_flat_size, const int _tens_sz): cpt(cpt), ginit(ginit), graph_flat_size(graph_flat_size), tens_sz(_tens_sz){
  }

  size_t outputLinearSize() const{ return graph_flat_size; }
  size_t inputLinearSize() const{ return graph_flat_size + tens_sz; }
  
  Matrix<FloatType> value(const Matrix<FloatType> &in, EnableDeriv enable_deriv = DerivNo){
    int batch_size = in.size(1);
    ginit.batch_size = batch_size;
    Graph<FloatType> graph(ginit);
    int poff = unflattenFromBatchVector(graph, in, 0);
    
    Tensor<FloatType,2> node_attr_update(tens_sz, batch_size);
    unflattenFromBatchVector(node_attr_update, in, poff);

    return flattenToBatchVector( cpt.value(graph, node_attr_update) );
  }
  void deriv(Vector<FloatType> &cost_deriv_params, int off, Matrix<FloatType> &&_above_deriv_lin, Matrix<FloatType> &cost_deriv_inputs){
    int batch_size = _above_deriv_lin.size(1);
    ginit.batch_size = batch_size;
    Graph<FloatType> above_deriv = unflattenFromBatchVector(_above_deriv_lin, ginit);
    
    Tensor<FloatType,2> cost_deriv_inputs_tens;
    Graph<FloatType> cost_deriv_inputs_graph;
    cpt.deriv(std::move(above_deriv), cost_deriv_inputs_graph, cost_deriv_inputs_tens);

    cost_deriv_inputs = Matrix<FloatType>(inputLinearSize(),batch_size);
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

void testInsertGlobalUpdateOutput(){
  std::mt19937 rng(1234);
  typedef confDouble Config;
  typedef double FloatType;

  GraphInitialize ginit;
  ginit.nnode = 3;
  ginit.node_attr_sizes = std::vector<int>({3,4});
  ginit.edge_attr_sizes = std::vector<int>({5,6});

  ginit.edge_map = std::vector<std::pair<int,int> >({  {0,1}, {1,2}, {2,0}, {1,0}, {2,1}, {0,2} });
  ginit.global_attr_sizes = std::vector<int>({2});
  ginit.batch_size =4;
  
  Graph<FloatType> graph(ginit);
  graph.applyToAllAttributes([&](Tensor<FloatType,3> &m){ uniformRandom(m,rng); });

  InsertGlobalUpdateOutputComponent<Config> gup_cpt;
  Tensor<FloatType,2> gup_in(ginit.totalAttribSize(ginit.global_attr_sizes), ginit.batch_size);
  uniformRandom(gup_in, rng);

  Graph<FloatType> gup = gup_cpt.value(graph, gup_in);
  Graph<FloatType> gexpect = expectInsertGlobalUpdateOutput(graph, gup_in);
  assert(equal(gup, gexpect,true));

  InsertGlobalUpdateOutputComponentWrapper<Config> wrp(gup_cpt, ginit, rowsAsBatchVector(graph), gup_in.size(0));
  testComponentDeriv(wrp, ginit.batch_size, 1e-4, true);
  testComponentDiffBatchSizes(wrp);
  
  std::cout << "testInsertGlobalUpdateOutput passed" << std::endl;
}

void testGlobalUpdateBlock(){
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

  //the model
  auto splt = replicate_layer(2, input_layer<Config, InputType>());
  auto eagg = edge_aggregate_global_sum_layer(*splt[0]);
  auto nagg = node_aggregate_global_sum_layer(eagg);
  auto gup_in = extract_global_update_input_layer(nagg);
  
  std::array<int,2> gup_in_tens_sz = ExtractGlobalUpdateInputComponent<Config>::outputTensorSize(ginit);
  std::array<int,2> gup_out_tens_sz = InsertGlobalUpdateOutputComponent<Config>::inputTensorSize(ginit);

  Matrix<FloatType> weight(gup_out_tens_sz[0], gup_in_tens_sz[0]);
  Vector<FloatType> bias(gup_out_tens_sz[0]);
  uniformRandom(weight);
  uniformRandom(bias);
  
  auto gup = batch_tensor_dnn_layer<2>(weight, bias, 0, noActivation<FloatType>(), gup_in );  
  auto gup_out = insert_global_update_output_layer(*splt[1], gup);

  //compute expectation
  Graph<FloatType> expect;
  {
    Graph<FloatType> eagg_exp = expectEdgeAggregateGlobalSum(graph);
    Graph<FloatType> nagg_exp = expectNodeAggregateGlobalSum(eagg_exp);
    Tensor<FloatType,2> tens_in = expectExtractGlobalUpdateInput(nagg_exp);
    Tensor<FloatType,2> tens_out = matrixBatchTensorAxpy(weight, tens_in, bias, 0);
    expect = expectInsertGlobalUpdateOutput(graph, tens_out);
  }
  Graph<FloatType> got = gup_out.value(graph, DerivNo);
  assert(abs_near(got,expect,FloatType(1e-5),true));
  
  GraphInGraphOutLayerWrapper<Config, decltype(gup_out)> wrp(gup_out, ginit, rowsAsBatchVector(graph));
  testComponentDeriv(wrp, ginit.batch_size, 1e-4, true);
  testComponentDiffBatchSizes(wrp);
        
  std::cout << "testGlobalUpdateBlock passed" << std::endl;
}
