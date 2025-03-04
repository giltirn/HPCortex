#include <HPCortex.hpp>

void testSimpleLinear(){  
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
    
  auto model = mse_cost( dnn_layer(input_layer<FloatType>(), winit, binit) );
  DecayScheduler<FloatType> lr(0.01, 0.1);
  GradientDescentOptimizer<FloatType, DecayScheduler<FloatType> > opt(lr);
  
  train(model, data, opt, 200, 1);

  std::cout << "Final params" << std::endl;
  Vector<FloatType> final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p(i) << std::endl;

}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinear();

  return 0;
}
