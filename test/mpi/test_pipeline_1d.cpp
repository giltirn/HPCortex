#include <HPCortex.hpp>
#include <Testing.hpp>

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
    if(rank == 0) if(ef != gf){ std::cout << "Value FLOPS mismatch, got " << gf << " expect " << ef << std::endl;  assert(0); }
    
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
  testPipelineLayer();
  return 0;
}
