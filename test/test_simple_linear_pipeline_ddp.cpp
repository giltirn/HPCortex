#include <HPCortex.hpp>
#include <Testing.hpp>

void testSimpleLinearPipelineDDP(){
  //Test f(x) = 0.2*x + 0.3;

  typedef float FloatType;
  
  //Setup pipeline groups of 2 ranks with DDP between them
  communicators().enableColorPipelining(communicators().worldRank()/2);
  communicators().reportSetup();
  
  int pipe_nranks = communicators().pipelineNrank();
  int pipe_rank = communicators().pipelineRank();

  int nranks_tot = communicators().worldNrank();

  int nepoch = 20;
  
  int call_batch_size = 2; //how many samples are processed at a time by each pipeline rank
  int in_out_dims[2] = {1, call_batch_size}; //all matrices here have the same dimension
  
  int glob_batch_size = 6 * nranks_tot; //how many samples in an overall batch
  int nbatch = 50; //how many batches in the data set
  
  int ddp_nbatch = communicators().ddpNrank(); //now many batches are computed in parallel under DDP
  
  int eff_batch_size = glob_batch_size *  ddp_nbatch; //effective batch size for reproduction on single rank
  
  int ndata = nbatch * glob_batch_size;
 
  std::vector<XYpair<FloatType,1,1> > data(ndata);

  for(int i=0;i<ndata;i++){
    data[i].x = Vector<FloatType>(1);
    data[i].y = Vector<FloatType>(1);
    
    FloatType eps = 2.0/(ndata - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1
    FloatType y = 0.2*x + 0.3;

    autoView(dxv,data[i].x,HostWrite);
    autoView(dyv,data[i].y,HostWrite);
    
    dxv(0) = x;
    dyv(0) = y;
  }
   
  Matrix<FloatType> winit(1,1,0.1);
  Vector<FloatType> binit(1,0.01);

  auto rank_model = pipe_rank == pipe_nranks-1 ? enwrap( dnn_layer(input_layer<FloatType>(), winit, binit) )  : enwrap( dnn_layer(input_layer<FloatType>(), winit, binit, ReLU<FloatType>()) );
 
  auto rank_block = pipeline_block< Matrix<FloatType>, Matrix<FloatType> >(rank_model, in_out_dims, in_out_dims);

  auto cost = BatchPipelineCostFuncWrapper<decltype(rank_block), MSEcostFunc<Matrix<FloatType> > >(rank_block, call_batch_size);

  auto full_model = enwrap( dnn_layer(input_layer<FloatType>(), winit, binit) );
  for(int i=0;i<pipe_nranks-1;i++)
    full_model = enwrap( dnn_layer(std::move(full_model), winit, binit, ReLU<FloatType>()) );
  auto full_cost = mse_cost(full_model);

  DecayScheduler<FloatType> lr(0.001, 0.1);
  AdamParams<FloatType> ap;
  AdamOptimizer<FloatType,DecayScheduler<FloatType> > opt(ap,lr);

  //Train pipeline + DDP
  train(cost, data, opt, nepoch, glob_batch_size);
  Vector<FloatType> final_p = cost.getParams();
  std::vector<Vector<FloatType> > predict(ndata);
  for(int i=0;i<ndata;i++) predict[i] = cost.predict(data[i].x);

  std::cout << "Training rank local model for comparison" << std::endl;  
  communicators().disableParallelism();
  communicators().reportSetup();
  train(full_cost, data, opt, nepoch, eff_batch_size, communicators().worldRank() != 0);
  Vector<FloatType> expect_p = full_cost.getParams();

  MPI_Barrier(MPI_COMM_WORLD);

  communicators().enableColorPipelining(communicators().worldRank()/2);
  
  if(communicators().ddpRank() == 0  && !pipe_rank){
    std::cout << "Final params " << final_p << " expect " << expect_p << std::endl;
    assert(near(final_p,expect_p,FloatType(1e-4),true));
    
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
