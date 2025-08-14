#include <HPCortex.hpp>
#include <Testing.hpp>

void testSimpleLinear(){  
  //Test f(x) = 0.2*x + 0.3;
  typedef confSingle Config;
  typedef typename Config::FloatType FloatType;

  int nepoch = 20;
  
  Matrix<FloatType> winit(1,1,0.0);
  Vector<FloatType> binit(1,0.0);

  int ndata_train = 100, ndata_valid=33;
  std::vector<XYpair<FloatType,1,1> > train_data(ndata_train);
  for(int i=0;i<ndata_train;i++){
    FloatType eps = 2.0/(ndata_train - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1

    train_data[i].x = Vector<FloatType>(1,x);
    train_data[i].y = Vector<FloatType>(1,0.2*x + 0.3);
  }

  std::vector<XYpair<FloatType,1,1> > valid_data(ndata_valid);
  for(int i=0;i<ndata_valid;i++){
    FloatType eps = 2.0/(ndata_valid - 1);
    FloatType x = -1.0 + i*eps;

    valid_data[i].x = Vector<FloatType>(1,x);
    valid_data[i].y = Vector<FloatType>(1,0.2*x + 0.3);
  }
  
  auto model = mse_cost( dnn_layer(winit, binit,input_layer<Config>()) );
  DecayScheduler<FloatType> lr(0.01, 0.1);
  GradientDescentOptimizer<FloatType, DecayScheduler<FloatType> > opt(lr);
  XYpairDataLoader<FloatType,1,1> train_loader(train_data);
  XYpairDataLoader<FloatType,1,1> valid_loader(valid_data);
  train(model, train_loader, valid_loader, opt, nepoch, 1);

  std::cout << "Final params" << std::endl;
  Vector<FloatType> final_p = model.getParams();
  autoView(final_p_v,final_p,HostRead);
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p_v(i) << std::endl;

  Vector<FloatType> expect_p(std::vector<FloatType>({0.2,0.3}));
  
  assert(near(expect_p,final_p,FloatType(1e-3),true));
  std::cout << "testSimpleLinear passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinear();

  return 0;
}
