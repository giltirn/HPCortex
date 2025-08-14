#include <HPCortex.hpp>
#include <Testing.hpp>

void testSimpleLinearDDP(){
  //Test f(x) = 0.2*x + 0.3;
  typedef confSingle Config;
  typedef float FloatType;
  
  Matrix<FloatType> winit(1,1,0.0);
  Vector<FloatType> binit(1,0.0);

  int nepoch = 10;
  
  int ndata = 100;
  std::vector<XYpair<FloatType,1,1> > data(ndata);
  for(int i=0;i<ndata;i++){
    FloatType eps = 2.0/(ndata - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1

    data[i].x = Vector<FloatType>(1,x);
    data[i].y = Vector<FloatType>(1,0.2*x + 0.3);
  }

  int ddp_eff_batch_size; //with DDP we are effectively increasing the batch size by nrank
  Vector<FloatType> params_global; //params from DDP training 
  {
    communicators().enableDDPnoPipelining();
    communicators().reportSetup();

    ddp_eff_batch_size = communicators().ddpNrank();
    
    auto model = mse_cost( dnn_layer(winit, binit,input_layer<Config>()) );
    DecayScheduler<FloatType> lr(0.01, 0.1);
    GradientDescentOptimizer<FloatType, DecayScheduler<FloatType> > opt(lr);
    XYpairDataLoader<FloatType,1,1> loader(data);
    train(model, loader, opt, nepoch, 1);
    params_global = model.getParams();
  }
  
  Vector<FloatType> params_local; //params from training on just this rank
  {
    communicators().disableParallelism();
    communicators().reportSetup();
    
    auto model = mse_cost( dnn_layer(winit, binit,input_layer<Config>()) );
    DecayScheduler<FloatType> lr(0.01, 0.1);
    GradientDescentOptimizer<FloatType, DecayScheduler<FloatType> > opt(lr);
    XYpairDataLoader<FloatType,1,1> loader(data);
    train(model, loader, opt, nepoch, ddp_eff_batch_size);
    params_local = model.getParams();
  }

  std::cout << "Params got " << params_global << " expect " << params_local << std::endl;
  assert(near(params_global, params_local, FloatType(1e-4), true));
  std::cout << "testSimpleLinearDDP passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinearDDP();

  return 0;
}
