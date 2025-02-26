#include <Layers.hpp>
#include <LossFunctions.hpp>
#include <Optimizers.hpp>  
#include <Comms.hpp>

void testSimpleLinear(){
  //Test f(x) = 0.2*x + 0.3;

  Matrix winit(1,1,0.0);
  Vector binit(1,0.0);

  int ndata = 100;
  std::vector<XYpair> data(ndata);
  for(int i=0;i<ndata;i++){
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1

    data[i].x = Vector(1,x);
    data[i].y = Vector(1,0.2*x + 0.3);
  }
    
  auto model = mse_cost( dnn_layer(input_layer(), winit, binit) );
  DecayScheduler lr(0.01, 0.1);
  GradientDescentOptimizer<DecayScheduler> opt(lr);
  
  train(model, data, opt, 200, 1);

  std::cout << "Final params" << std::endl;
  Vector final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p(i) << std::endl;

}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinear();

  return 0;
}
