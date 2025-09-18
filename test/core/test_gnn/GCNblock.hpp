void testGCNblock(){
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

  std::array<int,3> eup_in_tens_sz = ExtractEdgeUpdateInputComponent<Config>::outputTensorSize(ginit);
  std::array<int,3> eup_out_tens_sz = InsertEdgeUpdateOutputComponent<Config>::inputTensorSize(ginit);
  Matrix<FloatType> eup_weight(eup_out_tens_sz[1], eup_in_tens_sz[1]);
  Vector<FloatType> eup_bias(eup_out_tens_sz[1]);
  uniformRandom(eup_weight);
  uniformRandom(eup_bias);

  std::array<int,3> nup_in_tens_sz = ExtractNodeUpdateInputComponent<Config>::outputTensorSize(ginit);
  std::array<int,3> nup_out_tens_sz = InsertNodeUpdateOutputComponent<Config>::inputTensorSize(ginit);
  Matrix<FloatType> nup_weight(nup_out_tens_sz[1], nup_in_tens_sz[1]);
  Vector<FloatType> nup_bias(nup_out_tens_sz[1]);
  uniformRandom(nup_weight);
  uniformRandom(nup_bias);
  
  std::array<int,2> gup_in_tens_sz = ExtractGlobalUpdateInputComponent<Config>::outputTensorSize(ginit);
  std::array<int,2> gup_out_tens_sz = InsertGlobalUpdateOutputComponent<Config>::inputTensorSize(ginit);
  Matrix<FloatType> gup_weight(gup_out_tens_sz[0], gup_in_tens_sz[0]);
  Vector<FloatType> gup_bias(gup_out_tens_sz[0]);
  uniformRandom(gup_weight);
  uniformRandom(gup_bias);
  
  auto gcn_block = GCNblock(ginit,
			    [&](int fan_out, int fan_in, auto &&in){ //edge update
			      assert(eup_weight.size(0) == fan_out && eup_weight.size(1) == fan_in);
			      return batch_tensor_dnn_layer<3>(eup_weight, eup_bias, 1, noActivation<FloatType>(), std::forward<decltype(in)>(in) );
			    },
			    [&](int fan_out, int fan_in, auto &&in){ //node update
			      assert(nup_weight.size(0) == fan_out && nup_weight.size(1) == fan_in);
			      return batch_tensor_dnn_layer<3>(nup_weight, nup_bias, 1, noActivation<FloatType>(), std::forward<decltype(in)>(in) );
			    },
			    [&](int fan_out, int fan_in, auto &&in){ //global update
			      assert(gup_weight.size(0) == fan_out && gup_weight.size(1) == fan_in);
			      return batch_tensor_dnn_layer<2>(gup_weight, gup_bias, 0, noActivation<FloatType>(), std::forward<decltype(in)>(in) );
			    },
			    input_layer<Config, Graph<FloatType> >()
			    );
			    


  //compute expectation
  Graph<FloatType> expect;
  {
    //edge update
    Tensor<FloatType,3> eup_tens_in = expectExtractEdgeUpdateInput(graph);
    Tensor<FloatType,3> eup_tens_out = matrixBatchTensorAxpy(eup_weight, eup_tens_in, eup_bias, 1);
    Graph<FloatType> eup_out = expectInsertEdgeUpdateOutput(graph, eup_tens_out);

    //node update
    Graph<FloatType> eagg = expectEdgeAggregateSum(eup_out);
    Tensor<FloatType,3> nup_tens_in = expectExtractNodeUpdateInput(eagg);
    Tensor<FloatType,3> nup_tens_out = matrixBatchTensorAxpy(nup_weight, nup_tens_in, nup_bias, 1);
    Graph<FloatType> nup_out = expectInsertNodeUpdateOutput(eup_out, nup_tens_out);

    //global update
    Graph<FloatType> eagg_exp = expectEdgeAggregateGlobalSum(nup_out);
    Graph<FloatType> nagg_exp = expectNodeAggregateGlobalSum(eagg_exp);
    Tensor<FloatType,2> gup_tens_in = expectExtractGlobalUpdateInput(nagg_exp);
    Tensor<FloatType,2> gup_tens_out = matrixBatchTensorAxpy(gup_weight, gup_tens_in, gup_bias, 0);
    expect = expectInsertGlobalUpdateOutput(nup_out, gup_tens_out);
  }
  Graph<FloatType> got = gcn_block.value(graph, DerivNo);
  assert(abs_near(got,expect,FloatType(1e-5),true));
  
  GraphInGraphOutLayerWrapper<Config, decltype(gcn_block)> wrp(gcn_block, ginit, flatSize(graph));
  testComponentDeriv(wrp, 1e-4, true);
      
  std::cout << "testGCNblock passed" << std::endl;
}

