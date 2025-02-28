#include <HPCortex.hpp>

void testSimpleLinearPipeline(){
  //Test f(x) = 0.2*x + 0.3;
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
  communicators().reportSetup();
  
  int nranks = communicators().pipelineNrank();
  int rank = communicators().pipelineRank();

  int call_batch_size = 2;
  int glob_batch_size = 6 * nranks;

  int nbatch = 10;

  int ndata = nbatch * glob_batch_size;
  std::vector<XYpair> data(ndata);

  for(int i=0;i<ndata;i++){
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1
    double y = 0.2*x + 0.3;
    
    data[i].x = Vector(1,x);
    data[i].y = Vector(1,y);
  }
   
  Matrix winit(1,1,0.1);
  Vector binit(1,0.01);

  auto rank_model = rank == nranks-1 ? enwrap( dnn_layer(input_layer(), winit, binit) )  : enwrap( dnn_layer(input_layer(), winit, binit, ReLU()) );
 
  auto rank_block = pipeline_block(rank_model, call_batch_size, 1,1,1);

  auto cost = BatchPipelineCostFuncWrapper<decltype(rank_block), MSEcostFunc>(rank_block, call_batch_size);

  auto full_model = enwrap( dnn_layer(input_layer(), winit, binit) );
  for(int i=0;i<nranks-1;i++)
    full_model = enwrap( dnn_layer(std::move(full_model), winit, binit, ReLU()) );
  auto full_cost = mse_cost(full_model);

  DecayScheduler lr(0.001, 0.1);
  AdamParams ap;
  AdamOptimizer<DecayScheduler> opt(ap,lr);

  //Train pipeline
  train(cost, data, opt, 50, glob_batch_size);
  Vector final_p = cost.getParams();
  std::vector<Vector> predict(ndata);
  for(int i=0;i<ndata;i++) predict[i] = cost.predict(data[i].x);

  std::cout << "Training rank local model for comparison" << std::endl;  
  communicators().disableParallelism();
  communicators().reportSetup();
  train(full_cost, data, opt, 50, glob_batch_size);
  Vector expect_p = full_cost.getParams();

  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!rank){
    std::cout << "Final params " << final_p << " expect " << expect_p << std::endl;
    
    std::cout << "Predictions:" << std::endl;
    for(int i=0;i<nbatch;i++)
      std::cout << "Got " << predict[i] << " expect " << full_cost.predict(data[i].x) << " actual " << data[i].y << std::endl;
  }

}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinearPipeline();

  return 0;
}
