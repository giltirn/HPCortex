#include <Layers.hpp>
#include <LossFunctions.hpp>
#include <Optimizers.hpp>  
#include <Comms.hpp>

void testSimpleLinearDDP(){
  //Test f(x) = 0.2*x + 0.3;

  Matrix winit(1,1,0.0);
  Vector binit(1,0.0);

  int ndata = 100;
  std::vector<XYpair> data(ndata);
  for(int i=0;i<ndata;i++){
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1

    data[i].x = Matrix(1,1,x);
    data[i].y = Matrix(1,1,0.2*x + 0.3);
  }

  Vector params_local; //params from training on just this rank
  {
    communicators().disableParallelism();
    communicators().reportSetup();
    
    auto model = mse_cost( dnn_layer(input_layer(), winit, binit) );
    DecayScheduler lr(0.01, 0.1);
    GradientDescentOptimizer<DecayScheduler> opt(lr);
    
    train(model, data, opt, 200);
    params_local = model.getParams();
  }
  Vector params_global; //params from DDP training 
  {
    communicators().enableDDPnoPipelining();
    communicators().reportSetup();
    
    auto model = mse_cost( dnn_layer(input_layer(), winit, binit) );
    DecayScheduler lr(0.01, 0.1);
    GradientDescentOptimizer<DecayScheduler> opt(lr);
    
    train(model, data, opt, 200);
    params_global = model.getParams();
  }
  std::cout << "Params got " << params_global << " expect " << params_local << std::endl;
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinearDDP();

  return 0;
}
