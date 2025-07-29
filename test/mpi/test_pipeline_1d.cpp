#include <HPCortex.hpp>
#include <Testing.hpp>

void testPipeline(){
  typedef confSinglePipeline PipelineConfig;
  typedef confSingle StdConfig;
  typedef float FloatType;
  
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
    
  int nranks = communicators().pipelineNrank();
  int rank = communicators().pipelineRank();
  
  int batch_size = 1;
  int input_features = 1;
  int input_dims[2] = {input_features, batch_size};  

  FloatType B=0.15;
  FloatType A=3.14;
  
  Matrix<FloatType> winit(1,1,A);
  Vector<FloatType> binit(1,B);
  int block_output_dims[2] = {1, batch_size};

  if(1){ //test model
    if(!rank) std::cout << "Testing model value pipeline" << std::endl;

    auto p = pipeline_block< Matrix<FloatType>, Matrix<FloatType> >( dnn_layer(winit,binit,input_layer<PipelineConfig>()), block_output_dims, rank == nranks - 1  ? input_dims : block_output_dims);
    
    int value_lag = p.valueLag(); //iterations before first complete cycle of forwards differentiation
    int deriv_lag = p.derivLag(); //iterations before first complete cycle of backwards differentiation
    
    int iters=20;

    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<StdConfig>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(winit,binit,std::move(test_model)) ); 

    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<Matrix<FloatType> > expect_v(iters);
    std::vector<Vector<FloatType> > expect_d(iters, Vector<FloatType>(test_model.nparams()) );
    
    std::vector<Matrix<FloatType> > input_deriv(iters);
    for(int i=0;i<iters;i++){
      input_deriv[i] = Matrix<FloatType>(1,batch_size, 2.13*(i+1)); 
      Matrix<FloatType> x(1,1, i+1);
      expect_v[i] = test_model.value(x,DerivYes);

      Matrix<FloatType> idcp(input_deriv[i]);
      test_model.deriv(expect_d[i],0,std::move(idcp));
    }
    int nparams = test_model.nparams();

    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      Matrix<FloatType> x(1,1, i+1);
      Matrix<FloatType> v = p.value(x,DerivYes);
      Vector<FloatType> d(nparams,0.);

      int i_vpipe = i-(value_lag-1); //lag=3    2->0  3->1
      int i_dpipe = i-(deriv_lag-1);
      p.deriv(d,i_vpipe >= 0 ? input_deriv[i_vpipe] : Matrix<FloatType>(1,batch_size,-1)); //use the input deriv appropriate to the item index!
      
      if(!rank){

	if(i_vpipe >=0 ){
	  autoView(ev_i_v, expect_v[i_vpipe], HostRead);
	  autoView(v_v,v,HostRead);
	  
	  FloatType ev = ev_i_v(0,0); 
	  std::cout << i << "\tval expect " << ev << " got "<<  v_v(0,0) << std::endl;
	  assert(near(ev,v_v(0,0),FloatType(1e-4)));
	}
	if(i_dpipe >=0 ){
	  Vector<FloatType> ed = expect_d[i_dpipe];	
	  std::cout << "\tderiv expect " << ed << " got " << d << std::endl;
	  assert(near(d,ed,FloatType(1e-4),true));
	}
      }
    }
  }
  if(1){ //test cost
    if(!rank) std::cout << "Testing loss pipeline" << std::endl;
    auto p = pipeline_block< Matrix<FloatType>, Matrix<FloatType> >( dnn_layer(winit,binit,input_layer<PipelineConfig>()) , block_output_dims, rank == nranks - 1  ? input_dims : block_output_dims);
    PipelineCostFuncWrapper<decltype(p),MSEcostFunc<Matrix<FloatType>> > pc(p);
    int value_lag = p.valueLag();
    int deriv_lag = p.derivLag();
    
    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<StdConfig>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(winit,binit,std::move(test_model)) ); 
    auto test_cost = mse_cost(test_model);

    int nparams = p.nparams();
    
    int iters=20;

    std::vector<Matrix<FloatType> > x(iters);
    std::vector<Matrix<FloatType>> y(iters);
    
    for(int i=0;i<iters;i++){
      x[i] = Matrix<FloatType>(1,1, i+1);

      FloatType ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      y[i] = Matrix<FloatType>(1,1, 1.05*ival);
    }

    //Get expectation loss and derivatives
    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<FloatType> expect_l(iters);
    std::vector<Vector<FloatType> > expect_d(iters, Vector<FloatType>(test_model.nparams()) );
    for(int i=0;i<iters;i++){
      expect_l[i] = test_cost.loss(x[i],y[i],DerivYes);
      expect_d[i] = test_cost.deriv();
    }
    
    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      int i_vpipe = i-(value_lag-1);
      FloatType loss = pc.loss(x[i],y[i],DerivYes).first;
      FloatType loss_expect = i_vpipe < 0 ? -1. : expect_l[i_vpipe];

      int i_dpipe = i-(deriv_lag-1); //item index associated with derivative
      Vector<FloatType> deriv = pc.deriv().first;
      Vector<FloatType> deriv_expect = i_dpipe < 0 ? Vector<FloatType>(nparams,-1.) : expect_d[i_dpipe];
      
      if(!rank){
	std::cout << i << "\tvalue expect " << loss_expect << " got "<<  loss << std::endl;
	std::cout << "\tderiv expect " << deriv_expect << " got " << deriv << std::endl;
	assert(near(loss_expect,loss,FloatType(1e-4)));
	assert(near(deriv_expect,deriv,FloatType(1e-4),true));
      }
    }
  }


  if(1){ //test batched cost
    if(!rank) std::cout << "Testing batch loss pipeline" << std::endl;

    int glob_batch_size = 6*nranks;
    int call_batch_size = 2;

    int input_dims_2[2] = {input_features, call_batch_size};  
    int block_output_dims_2[2] = {1, call_batch_size};
    
    auto p = pipeline_block<Matrix<FloatType>, Matrix<FloatType> >( dnn_layer(winit,binit,input_layer<PipelineConfig>()) , block_output_dims_2, rank == nranks -1 ? input_dims_2 : block_output_dims_2);
    BatchPipelineCostFuncWrapper<decltype(p),MSEcostFunc<Matrix<FloatType>> > pc(p, call_batch_size);

    Matrix<FloatType> x(input_features, glob_batch_size);
    Matrix<FloatType> y(1, glob_batch_size);

    for(int i=0;i<glob_batch_size;i++){
      pokeColumn(x,i,Vector<FloatType>(1,i+1));

      FloatType ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      pokeColumn(y, i, Vector<FloatType>(1, 1.05*ival) );
    }
    
    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<StdConfig>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(winit,binit,std::move(test_model)) ); 
    auto test_cost = mse_cost(test_model);


    FloatType loss_expect = test_cost.loss(x,y,DerivYes);
    Vector<FloatType> deriv_expect = test_cost.deriv();

    FloatType loss_got = pc.loss(x,y,DerivYes);
    Vector<FloatType> deriv_got = pc.deriv();

    if(!rank){
      std::cout << "Loss - got " << loss_got << " expect " << loss_expect << std::endl;
      std::cout << "Deriv - got " << deriv_got << " expect " << deriv_expect << std::endl;
      assert(near(loss_expect,loss_got,FloatType(1e-4)));
      assert(near(deriv_expect,deriv_got,FloatType(1e-4),true));
    }
  }
  std::cout << "testPipeline passed" << std::endl;
}

void testPipelineLayer(){
  std::mt19937 rng(1234);

  typedef ModelConfiguration<float, FillEmptyRingBuffer> pconfSingle;
  typedef ModelConfiguration<double, FillEmptyRingBuffer> pconfDouble;
  
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
  assert(communicators().pipelineNrank() > 1);
  int rank = communicators().pipelineRank();
  int nrank = communicators().pipelineNrank();
  
  //Check we can wrap a layer and correctly call value
  {
    int dim[2] = {2,3};
    Tensor<float,2> tens(dim);
    uniformRandom(tens,rng);    
    
    LayerIOcontainer tc(tens);

    auto layer = dnn_layer(4,2,			 
			   input_layer<confSingle>()
			   );
    PipelineBlockContainer<LeafRef<decltype(layer)> > con(layer);
  
    LayerIOcontainer got = con.blockValue(tc,DerivNo);
    auto expect = layer.value(tens);
    assert(equal(got.as<Tensor<float,2> >(), expect,true));
  }
  //Check comms of io container wrapper
  {
    std::cout << "Checking comms of io container wrapper" << std::endl;
    LayerIOcontainer tc;
    if(rank == 0){
      int dim[2] = {2,3};
      Tensor<float,2> tens(dim);
      doHost(tens,{
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      tens_v(i,j) = j+3*i;
	});
      tc.insert(std::move(tens));
    }else if(rank == 1){
      tc.setType< Tensor<float,2> >();
    }
    std::vector<CommsRequest> reqs;
    pipelineSendRecvInitializer(reqs, tc, tc, 1, 0);
    waitAll(reqs);
    if(rank == 1){
      int const* sizes = tc.as< Tensor<float,2> >().sizeArray();
      assert(sizes[0] == 2 && sizes[1] == 3);
    }
    pipelineSendRecv(reqs, tc, tc, 1, 0);
    waitAll(reqs);
    if(rank == 1){
      int dim[2] = {2,3};
      Tensor<float,2> expect(dim);
      doHost(expect,{
	  for(int i=0;i<2;i++)
	    for(int j=0;j<3;j++)
	      expect_v(i,j) = j+3*i;
	});
      assert(equal(expect, tc.as< Tensor<float,2> >(), true));      
    
      //check copy, move
      {
	LayerIOcontainer tc_cp(tc);
	assert(equal(tc.as<Tensor<float,2>>(), tc_cp.as<Tensor<float,2>>(), true));

	LayerIOcontainer tc_mv(std::move(tc_cp));
	assert(equal(tc.as<Tensor<float,2>>(), tc_mv.as<Tensor<float,2>>(), true));

	LayerIOcontainer tc_mv2;
	tc_mv2 = std::move(tc_mv);
	assert(equal(tc.as<Tensor<float,2>>(), tc_mv2.as<Tensor<float,2>>(), true));
      }	
      
      //check remove
      Tensor<float,2> rm = tc.remove< Tensor<float,2> >();
      assert(equal(expect, rm, true));
    } //rank==1
  }

    
  //Check layer functionality
  {
    std::cout << "Checking layer functionality" << std::endl;
    int ubatch_size = 2;
    int batch_size = nrank * ubatch_size * 3;

    int fan_in = 3;
    int fan_out = 3;

    //have a non-trivial model below
    Matrix<double> bweight(fan_out,fan_in);
    Vector<double> bbias(fan_out);
    uniformRandom(bweight,rng);
    uniformRandom(bbias,rng);
    
    auto pbelow = dnn_layer(bweight, bbias, input_layer<pconfDouble>());
    // PipelineBlockLayer< Matrix<double>, LeafRef<decltype(pbelow)> > player(pbelow, ubatch_size);
    auto player = pipeline_block_layer<Matrix<double> >(ubatch_size, pbelow);
    
    //For covenience use a uniform fan_in, fan_out
    std::vector<Matrix<double> > weights(nrank, Matrix<double>(fan_out,fan_in));
    std::vector<Vector<double> > biases(nrank, Vector<double>(fan_out));
    for(int r=0;r<nrank;r++){ 
      uniformRandom(weights[r],rng);
      uniformRandom(biases[r],rng);
    }
      
    player.setRankBlock(dnn_layer(weights[rank],biases[rank], input_layer<pconfDouble>()));

    //have a non-trivial model above too
    Matrix<double> aweight(fan_out,fan_in);
    Vector<double> abias(fan_out);
    uniformRandom(aweight,rng);
    uniformRandom(abias,rng);
    auto got_model = dnn_layer(aweight,abias, player);
    
    //generate the equivalent model on each rank separately
    auto expect_model = enwrap(dnn_layer(bweight,bbias,input_layer<confDouble>()));
    for(int r=0;r<nrank;r++)
      expect_model = enwrap(dnn_layer(weights[r],biases[r], std::move(expect_model)));
    expect_model = enwrap(dnn_layer(aweight,abias,std::move(expect_model)));
    
    assert(got_model.nparams() == expect_model.nparams());
    
    for(int i=0;i<2;i++){ //run twice to ensure consistency as we store initializers on the first call
      Matrix<double> input(fan_in, batch_size);
      uniformRandom(input, rng);
      
      //value
      Matrix<double> expect = expect_model.value(input,DerivYes);
      Matrix<double> got = got_model.value(input,DerivYes);
      if(rank == 0) assert(abs_near(expect,got,1e-6,true));

      //deriv
      Matrix<double> above_deriv(fan_out, batch_size);
      uniformRandom(above_deriv,rng);
      
      Vector<double> got_der(got_model.nparams(),0.);
      Matrix<double> got_in_der(fan_in, batch_size);

      Vector<double> expect_der(expect_model.nparams(),0.);
      Matrix<double> expect_in_der(fan_in, batch_size);

      int dout_got = got_model.deriv(got_der, 0, Matrix<double>(above_deriv), &got_in_der);
      int dout_expect = expect_model.deriv(expect_der, 0, Matrix<double>(above_deriv), &expect_in_der);
      std::cout << "Offset got " << dout_got << " expect " << dout_expect << std::endl;
      assert(dout_got == dout_expect);

      if(rank == 0){
	std::cout << "Got der:\n" << got_der << "\nExpect der:\n" << expect_der << std::endl;
	std::cout << "Got input der:\n" << got_in_der << "\nExpect in der:\n" << expect_in_der << std::endl;

	assert(near(got_der,expect_der,1e-6,true));
	assert(near(got_in_der,expect_in_der,1e-6,true));
      }
    }

    //check update
    Vector<double> new_params(got_model.nparams());
    uniformRandom(new_params,rng);
    Vector<double> dummy_params(got_model.nparams(), 0.);
    
    int poff = got_model.update(0, rank == 0 ? new_params : dummy_params); //ensure params are passed from rank 0
    int eoff = expect_model.update(0, new_params);
    assert(poff == eoff);

    Matrix<double> input(fan_in, batch_size);
    uniformRandom(input, rng);
     
    Matrix<double> expect = expect_model.value(input);
    Matrix<double> got = got_model.value(input);
    if(rank == 0) assert(abs_near(expect,got,1e-6,true));

    //check step
    poff = got_model.step(0, rank == 0 ? new_params : dummy_params, 0.567); //ensure params are passed from rank 0
    eoff = expect_model.step(0, new_params, 0.567);
    assert(poff == eoff);
    
    expect = expect_model.value(input);
    got = got_model.value(input);
    if(rank == 0) assert(abs_near(expect,got,1e-6,true));

    //check getparams
    got_model.update(0, rank == 0 ? new_params : dummy_params); //ensure params are passed from rank 0
   
    Vector<double> pgot_params(got_model.nparams());
    poff = got_model.getParams(pgot_params, 0);
    assert(poff == eoff);
    if(rank == 0) assert(equal(pgot_params, new_params, true));
    
    //check FLOPS
    size_t ef = expect_model.FLOPS(0);
    size_t gf = got_model.FLOPS(0);
    if(rank == 0) assert(ef == gf);
    
    ef = expect_model.FLOPS(1);
    gf = got_model.FLOPS(1);
    if(rank == 0) assert(ef == gf);

    
  }

  //Demonstrate you can chain pipeline layers in the normal way
  {
    std::cout << "Checking pipeline layer chaining" << std::endl;
    int ubatch_size = 2;
    int batch_size = nrank * ubatch_size * 3;

    int fan_in = 3;
    int fan_out = 3;

    //have a non-trivial model below
    Matrix<double> bweight(fan_out,fan_in);
    Vector<double> bbias(fan_out);
    uniformRandom(bweight,rng);
    uniformRandom(bbias,rng);
    
    auto pbelow = dnn_layer(bweight, bbias, input_layer<pconfDouble>());
    auto player1 = pipeline_block_layer<Matrix<double> >(ubatch_size, pbelow);
    auto player2 = pipeline_block_layer<Matrix<double> >(ubatch_size, player1);
    //PipelineBlockLayer< Matrix<double>, LeafRef<decltype(pbelow)> > player1(pbelow, ubatch_size);
    //PipelineBlockLayer< Matrix<double>, LeafRef<decltype(player1)> > player2(player1, ubatch_size);

    //For covenience use a uniform fan_in, fan_out
    std::vector<Matrix<double> > weights1(nrank, Matrix<double>(fan_out,fan_in)), weights2(nrank, Matrix<double>(fan_out,fan_in));
    std::vector<Vector<double> > biases1(nrank, Vector<double>(fan_out)), biases2(nrank, Vector<double>(fan_out));
    for(int r=0;r<nrank;r++){ 
      uniformRandom(weights1[r],rng);
      uniformRandom(biases1[r],rng);

      uniformRandom(weights2[r],rng);
      uniformRandom(biases2[r],rng);
    }
      
    player1.setRankBlock(dnn_layer(weights1[rank],biases1[rank], input_layer<pconfDouble>()));
    player2.setRankBlock(dnn_layer(weights2[rank],biases2[rank], input_layer<pconfDouble>()));
    
    //have a non-trivial model above too
    Matrix<double> aweight(fan_out,fan_in);
    Vector<double> abias(fan_out);
    uniformRandom(aweight,rng);
    uniformRandom(abias,rng);
    auto got_model = dnn_layer(aweight,abias, player2);
    
    //generate the equivalent model on each rank separately
    auto expect_model = enwrap(dnn_layer(bweight,bbias,input_layer<confDouble>()));
    for(int r=0;r<nrank;r++)
      expect_model = enwrap(dnn_layer(weights1[r],biases1[r], std::move(expect_model)));
    for(int r=0;r<nrank;r++)
      expect_model = enwrap(dnn_layer(weights2[r],biases2[r], std::move(expect_model)));
    expect_model = enwrap(dnn_layer(aweight,abias,std::move(expect_model)));
    
    assert(got_model.nparams() == expect_model.nparams());
    
    for(int i=0;i<2;i++){ //run twice to ensure consistency as we store initializers on the first call
      Matrix<double> input(fan_in, batch_size);
      uniformRandom(input, rng);
      
      //value
      Matrix<double> expect = expect_model.value(input,DerivYes);
      Matrix<double> got = got_model.value(input,DerivYes);
      if(rank == 0) assert(abs_near(expect,got,1e-6,true));

      //deriv
      Matrix<double> above_deriv(fan_out, batch_size);
      uniformRandom(above_deriv,rng);
      
      Vector<double> got_der(got_model.nparams(),0.);
      Matrix<double> got_in_der(fan_in, batch_size);

      Vector<double> expect_der(expect_model.nparams(),0.);
      Matrix<double> expect_in_der(fan_in, batch_size);

      int dout_got = got_model.deriv(got_der, 0, Matrix<double>(above_deriv), &got_in_der);
      int dout_expect = expect_model.deriv(expect_der, 0, Matrix<double>(above_deriv), &expect_in_der);
      std::cout << "Offset got " << dout_got << " expect " << dout_expect << std::endl;
      assert(dout_got == dout_expect);

      if(rank == 0){
	std::cout << "Got der:\n" << got_der << "\nExpect der:\n" << expect_der << std::endl;
	std::cout << "Got input der:\n" << got_in_der << "\nExpect in der:\n" << expect_in_der << std::endl;

	assert(near(got_der,expect_der,1e-6,true));
	assert(near(got_in_der,expect_in_der,1e-6,true));
      }
    }

    //check update
    Vector<double> new_params(got_model.nparams());
    uniformRandom(new_params,rng);
    Vector<double> dummy_params(got_model.nparams(), 0.);
    
    int poff = got_model.update(0, rank == 0 ? new_params : dummy_params); //ensure params are passed from rank 0
    int eoff = expect_model.update(0, new_params);
    assert(poff == eoff);

    Matrix<double> input(fan_in, batch_size);
    uniformRandom(input, rng);
     
    Matrix<double> expect = expect_model.value(input);
    Matrix<double> got = got_model.value(input);
    if(rank == 0) assert(abs_near(expect,got,1e-6,true));

    //check step
    poff = got_model.step(0, rank == 0 ? new_params : dummy_params, 0.567); //ensure params are passed from rank 0
    eoff = expect_model.step(0, new_params, 0.567);
    assert(poff == eoff);
    
    expect = expect_model.value(input);
    got = got_model.value(input);
    if(rank == 0) assert(abs_near(expect,got,1e-6,true));

    //check getparams
    got_model.update(0, rank == 0 ? new_params : dummy_params); //ensure params are passed from rank 0
   
    Vector<double> pgot_params(got_model.nparams());
    poff = got_model.getParams(pgot_params, 0);
    assert(poff == eoff);
    if(rank == 0) assert(equal(pgot_params, new_params, true));
    
    //check FLOPS
    size_t ef = expect_model.FLOPS(0);
    size_t gf = got_model.FLOPS(0);
    if(rank == 0) assert(ef == gf);
    
    ef = expect_model.FLOPS(1);
    gf = got_model.FLOPS(1);
    if(rank == 0) assert(ef == gf);

    
  }

  

}






int main(int argc, char** argv){
  initialize(argc, argv);

  //testPipeline();
  testPipelineLayer();
  return 0;
}
