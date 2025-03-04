#include <HPCortex.hpp>

void testSimpleLinearDDP(){
  //Test f(x) = 0.2*x + 0.3;

  typedef float FloatType;
  
  Matrix<FloatType> winit(1,1,0.0);
  Vector<FloatType> binit(1,0.0);

  int ndata = 100;
  std::vector<XYpair<FloatType> > data(ndata);
  for(int i=0;i<ndata;i++){
    FloatType eps = 2.0/(ndata - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1

    data[i].x = Vector<FloatType>(1,x);
    data[i].y = Vector<FloatType>(1,0.2*x + 0.3);
  }

  Vector<FloatType> params_local; //params from training on just this rank
  {
    communicators().disableParallelism();
    communicators().reportSetup();
    
    auto model = mse_cost( dnn_layer(input_layer<FloatType>(), winit, binit) );
    DecayScheduler<FloatType> lr(0.01, 0.1);
    GradientDescentOptimizer<FloatType, DecayScheduler<FloatType> > opt(lr);
    
    train(model, data, opt, 200, 1);
    params_local = model.getParams();
  }
  Vector<FloatType> params_global; //params from DDP training 
  {
    communicators().enableDDPnoPipelining();
    communicators().reportSetup();
    
    auto model = mse_cost( dnn_layer(input_layer<FloatType>(), winit, binit) );
    DecayScheduler<FloatType> lr(0.01, 0.1);
    GradientDescentOptimizer<FloatType, DecayScheduler<FloatType> > opt(lr);
    
    train(model, data, opt, 200, 1);
    params_global = model.getParams();
  }
  std::cout << "Params got " << params_global << " expect " << params_local << std::endl;
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinearDDP();

  return 0;
}
