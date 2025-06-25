#include <HPCortex.hpp>
#include <Testing.hpp>

void testSimpleLinear2D(){  
  //Test f(x,y) = { 0.2*x + 0.3, -0.1*y - 0.3 }   using a Tensor<2> data type
  typedef float FloatType;
  typedef std::vector<FloatType> vec;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<FloatType> udist(0.0,1.0);
  
  int nepoch = 100;
  
  Matrix<FloatType> winit(2,2,0.0);
  Vector<FloatType> binit(2,0.0);

  int ndata = 500;

  //To avoid large off-diagonal elements in the weight matrix, we want to decorrelate the two dimensions
  std::vector<XYpair<FloatType,2,2> > data(ndata);
  int tens_sz[2] = {2,1};
  for(int i=0;i<ndata;i++){
    FloatType x = udist(rng);
    FloatType y = udist(rng);
    
    data[i].x = Matrix<FloatType>(tens_sz);
    data[i].y = Matrix<FloatType>(tens_sz);
    autoView(x_v,data[i].x,HostWrite);
    autoView(y_v,data[i].y,HostWrite);
    
    {
      x_v(0,0) = x;
      x_v(1,0) = y;
      y_v(0,0) = 0.2*x + 0.3;
      y_v(1,0) = -0.1*y - 0.3;
    }
  }

  int batch_size = 32;
  int tens_sz_batch[3] = {2,1,batch_size};
  
  //Flatten and unflatten to push through a DNN layer
  auto model = mse_cost(
			unflatten_layer<3>(tens_sz_batch,
					   dnn_layer(winit, binit,
						     flatten_layer(
								   input_layer<FloatType, Tensor<FloatType,3> >()
								   )
						     )					
					   )
			);
  DecayScheduler<FloatType> lr(0.01, 0.1);  
  AdamParams<FloatType> ap;
  AdamOptimizer<FloatType, DecayScheduler<FloatType> > opt(ap, lr);
  
  train(model, data, opt, nepoch, batch_size);

  std::cout << "Final params" << std::endl;
  Vector<FloatType> final_p = model.getParams();
  autoView(final_p_v,final_p,HostRead);
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p_v(i) << std::endl;

  int ms[2] = {2,2};
  int vs[1] = {2};

  Matrix<FloatType> wexpect(ms, vec({0.2,0,0,-0.1}));
  Vector<FloatType> bexpect(vs, vec({0.3,-0.3}));
  Matrix<FloatType> wgot;
  Vector<FloatType> bgot;
  
  doHost(final_p, {  
      wgot = Matrix<FloatType>(ms, final_p_v.data());
      bgot = Vector<FloatType>(vs, final_p_v.data()+4);
    });
  
  std::cout << "Got weight " << wgot << std::endl << "Expect " << wexpect << std::endl;
  std::cout << "Got bias " << bgot << std::endl << "Expect " << bexpect << std::endl;

  assert(abs_near(wgot,wexpect,FloatType(1e-3)));
  assert(abs_near(bgot,bexpect,FloatType(1e-3)));
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinear2D();

  return 0;
}
