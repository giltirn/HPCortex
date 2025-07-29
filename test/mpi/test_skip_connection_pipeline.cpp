#include <HPCortex.hpp>
#include <Testing.hpp>

//Note, only skipping within a rank, not between!
void testSkipConnectionPipeline(){
  std::mt19937 rng(1234);

  typedef confDoublePipelineNew PipelineConfig;
  typedef confDouble StdConfig;
  typedef double FloatType;
  
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
    
  int nranks = communicators().pipelineNrank();
  int rank = communicators().pipelineRank();
  
  int ubatch_size = 2;
  int glob_batch_size = nranks * ubatch_size;
   
  FloatType A=3.14;
  FloatType B=0.15;
  FloatType C=5.66;
  FloatType D=-3.455;
  
  Matrix<FloatType> winit1(1,1,A);
  Vector<FloatType> binit1(1,B);
  Matrix<FloatType> winit2(1,1,C);
  Vector<FloatType> binit2(1,D);
  
  if(!rank) std::cout << "Testing model value pipeline" << std::endl;
  auto skip1 = skip_connection( dnn_layer(winit1,binit1,input_layer<PipelineConfig>()), input_layer<PipelineConfig>());
  auto skip2 = skip_connection( dnn_layer(winit2,binit2,input_layer<PipelineConfig>()), skip1);

  auto player = pipeline_block_layer<Matrix<FloatType> >(ubatch_size, input_layer<PipelineConfig>());
  player.setRankBlock(skip2);

  //Build the same model on just this rank
  auto test_model = enwrap( input_layer<StdConfig>() );
  for(int r=0;r<nranks;r++){
    test_model = enwrap( skip_connection( dnn_layer(winit1,binit1, input_layer<StdConfig>()), std::move(test_model) ) ); 
    test_model = enwrap( skip_connection( dnn_layer(winit2,binit2, input_layer<StdConfig>()), std::move(test_model) ) ); 
  }
      
  if(!rank) std::cout << "Computing expectations" << std::endl;
  Matrix<FloatType> x(1,glob_batch_size), above_deriv(1,glob_batch_size);
  uniformRandom(x,rng);
  uniformRandom(above_deriv,rng);
      
  Matrix<FloatType> expect_v = test_model.value(x,DerivYes);

  Vector<FloatType> expect_d(test_model.nparams());
  Matrix<FloatType> expect_id(1,glob_batch_size);
  test_model.deriv(expect_d,0, Matrix<FloatType>(above_deriv), &expect_id);

  Matrix<FloatType> got_v = player.value(x,DerivYes);

  Vector<FloatType> got_d(player.nparams());
  Matrix<FloatType> got_id(1,glob_batch_size);
  player.deriv(got_d,0, Matrix<FloatType>(above_deriv), &got_id);

  if(!rank){
    assert(near(expect_v,got_v,1e-5,true));
    assert(near(expect_d,got_d,1e-5,true));
    assert(near(expect_id,got_id,1e-5,true));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "testSkipConnectionPipeline passed" << std::endl;
}


int main(int argc, char** argv){
  initialize(argc, argv);

  testSkipConnectionPipeline();

  return 0;
}

