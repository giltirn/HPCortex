#include <HPCortex.hpp>
#include <Testing.hpp>

void testSimpleLinearPipeline(){
  //Test f(x) = 0.2*x + 0.3;
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
  communicators().reportSetup();
  
  int nranks = communicators().pipelineNrank();
  int rank = communicators().pipelineRank();

  int call_batch_size = 2;  
  int glob_batch_size = 6 * nranks;

  int nepoch = 10;
  int nbatch = 10;

  typedef confSinglePipelineNew PipelineConfig;
  typedef confSingle StdConfig;
  typedef float FloatType;
    
  int ndata = nbatch * glob_batch_size;
  std::vector<XYpair<FloatType,1,1> > data(ndata);

  for(int i=0;i<ndata;i++){
    FloatType eps = 2.0/(ndata - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1
    FloatType y = 0.2*x + 0.3;
    
    data[i].x = Vector<FloatType>(1,x);
    data[i].y = Vector<FloatType>(1,y);
  }
   
  Matrix<FloatType> winit(1,1,0.1);
  Vector<FloatType> binit(1,0.01);

  auto player = pipeline_block_layer< Matrix<FloatType> >(call_batch_size, input_layer<PipelineConfig>());
  if(rank == 0)
    player.setRankBlock(dnn_layer(winit, binit, input_layer<PipelineConfig>()));
  else
    player.setRankBlock(dnn_layer(winit, binit, ReLU<FloatType>(),input_layer<PipelineConfig>()));

  auto cost = mse_cost(player);

  auto full_model = enwrap( dnn_layer(winit, binit,input_layer<StdConfig>()) );
  for(int i=0;i<nranks-1;i++)
    full_model = enwrap( dnn_layer(winit, binit, ReLU<FloatType>(), std::move(full_model)) );
  auto full_cost = mse_cost(full_model);

  DecayScheduler<FloatType> lr(0.001, 0.1);
  AdamParams<FloatType> ap;
  AdamOptimizer<FloatType,DecayScheduler<FloatType> > opt(ap,lr);

  //Train pipeline
  XYpairDataLoader<FloatType,1,1> loader(data);
  train(cost, loader, opt, nepoch, glob_batch_size);
  Vector<FloatType> final_p = cost.getParams();
  std::vector<Vector<FloatType>> predict(ndata);
  for(int i=0;i<ndata;i++) predict[i] = cost.predict(data[i].x, glob_batch_size);

  std::cout << "Training rank local model for comparison" << std::endl;  
  communicators().disableParallelism();
  communicators().reportSetup();
  train(full_cost, loader, opt, nepoch, glob_batch_size);
  Vector<FloatType> expect_p = full_cost.getParams();

  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!rank){
    std::cout << "Final params " << final_p << " expect " << expect_p << std::endl;
    assert(near(final_p,expect_p,FloatType(1e-4),true));
    
    std::cout << "Predictions:" << std::endl;
    for(int i=0;i<nbatch;i++)
      std::cout << "Got " << predict[i] << " expect " << full_cost.predict(data[i].x, glob_batch_size) << " actual " << data[i].y << std::endl;
  }
  std::cout << "testSimpleLinearPipeline passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinearPipeline();

  return 0;
}
