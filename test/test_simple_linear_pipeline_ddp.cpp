#include <HPCortex.hpp>

void testSimpleLinearPipelineDDP(){
  //Test f(x) = 0.2*x + 0.3;

  typedef float FloatType;
  
  //Setup pipeline groups of 2 ranks with DDP between them
  communicators().enableColorPipelining(communicators().worldRank()/2);
  communicators().reportSetup();
  
  int pipe_nranks = communicators().pipelineNrank();
  int pipe_rank = communicators().pipelineRank();

  int nranks_tot = communicators().worldNrank();
  
  int call_batch_size = 2; //how many samples are processed at a time by each pipeline rank
  int glob_batch_size = 6 * nranks_tot; //how many samples in an overall batch

  int nbatch = 50; //how many batches in the data set

  int ndata = nbatch * glob_batch_size;
 
  std::vector<XYpair<FloatType> > data(ndata);

  for(int i=0;i<ndata;i++){
    data[i].x = Vector<FloatType>(1);
    data[i].y = Vector<FloatType>(1);
    
    FloatType eps = 2.0/(ndata - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1
    FloatType y = 0.2*x + 0.3;

    data[i].x(0) = x;
    data[i].y(0) = y;
  }
   
  Matrix<FloatType> winit(1,1,0.1);
  Vector<FloatType> binit(1,0.01);

  auto rank_model = pipe_rank == pipe_nranks-1 ? enwrap( dnn_layer(input_layer<FloatType>(), winit, binit) )  : enwrap( dnn_layer(input_layer<FloatType>(), winit, binit, ReLU<FloatType>()) );
 
  auto rank_block = pipeline_block(rank_model, call_batch_size, 1,1,1);

  auto cost = BatchPipelineCostFuncWrapper<FloatType,decltype(rank_block), MSEcostFunc<FloatType> >(rank_block, call_batch_size);

  auto full_model = enwrap( dnn_layer(input_layer<FloatType>(), winit, binit) );
  for(int i=0;i<pipe_nranks-1;i++)
    full_model = enwrap( dnn_layer(std::move(full_model), winit, binit, ReLU<FloatType>()) );
  auto full_cost = mse_cost(full_model);

  DecayScheduler<FloatType> lr(0.001, 0.1);
  AdamParams<FloatType> ap;
  AdamOptimizer<FloatType,DecayScheduler<FloatType> > opt(ap,lr);

  //Train pipeline
  train(cost, data, opt, 50, glob_batch_size);
  Vector<FloatType> final_p = cost.getParams();
  std::vector<Vector<FloatType> > predict(ndata);
  for(int i=0;i<ndata;i++) predict[i] = cost.predict(data[i].x);

  std::cout << "Training rank local model for comparison" << std::endl;  
  communicators().disableParallelism();
  communicators().reportSetup();
  train(full_cost, data, opt, 50, glob_batch_size);
  Vector<FloatType> expect_p = full_cost.getParams();

  MPI_Barrier(MPI_COMM_WORLD);

  communicators().enableColorPipelining(communicators().worldRank()/2);
  
  if(communicators().ddpRank() == 0  && !pipe_rank){
    std::cout << "Final params " << final_p << " expect " << expect_p << std::endl;
    
    std::cout << "Predictions:" << std::endl;
    for(int i=0;i<ndata;i++)
      std::cout << "Got " << predict[i] << " expect " << full_cost.predict(data[i].x) << " actual " << data[i].y << std::endl;
  }

}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinearPipelineDDP();

  return 0;
}
