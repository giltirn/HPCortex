#include <HPCortex.hpp>
#include <Testing.hpp>

void testSimpleLinear(){  
  //Test f(x) = 0.2*x + 0.3;
  typedef float FloatType;

  int nepoch = 20;
  
  Matrix<FloatType> winit(1,1,0.0);
  Vector<FloatType> binit(1,0.0);

  int ndata = 100;
  std::vector<XYpair<FloatType,1,1> > data(ndata);
  for(int i=0;i<ndata;i++){
    FloatType eps = 2.0/(ndata - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1

    data[i].x = Vector<FloatType>(1,x);
    data[i].y = Vector<FloatType>(1,0.2*x + 0.3);
  }
    
  auto model = mse_cost( dnn_layer(input_layer<FloatType>(), winit, binit) );
  DecayScheduler<FloatType> lr(0.01, 0.1);
  GradientDescentOptimizer<FloatType, DecayScheduler<FloatType> > opt(lr);
  
  train(model, data, opt, nepoch, 1);

  std::cout << "Final params" << std::endl;
  Vector<FloatType> final_p = model.getParams();
  autoView(final_p_v,final_p,HostRead);
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p_v(i) << std::endl;

  Vector<FloatType> expect_p(std::vector<FloatType>({0.2,0.3}));
  
  assert(near(expect_p,final_p,FloatType(1e-3),true));
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinear();

  return 0;
}
