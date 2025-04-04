#include <HPCortex.hpp>
#include <Testing.hpp>

void testOneHiddenLayer(){
  //Test f(x) = 0.2*x + 0.3;
  int nbatch = 100;
  int batch_size = 4;

  typedef float FloatType;
  FloatType delta = 1e-4;

  int nepoch = 20;
  int ndata = batch_size * nbatch;

  std::vector<XYpair<FloatType> > data(ndata);
  std::vector<int> didx(ndata);
  
  for(int i=0;i<ndata;i++){ //i = b + batch_size * B
    FloatType eps = 2.0/(ndata - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1

    data[i].x = Vector<FloatType>(1,x);
    data[i].y = Vector<FloatType>(1,0.2*x + 0.3);

    didx[i] = i;
  }

  int nhidden = 5;

  Matrix<FloatType> winit_out(1,nhidden,0.01);
  Matrix<FloatType> winit_h(nhidden,1,0.01);

  Vector<FloatType> binit_out(1,0.01);
  Vector<FloatType> binit_h(nhidden, 0.01);

  auto hidden_layer( dnn_layer(input_layer<FloatType>(), winit_h, binit_h, ReLU<FloatType>()) );
  auto model = mse_cost( dnn_layer(hidden_layer, winit_out, binit_out) );

  //Test derivative
  {
    Vector<FloatType> p = model.getParams();
    
    for(int d=1;d<5;d++){ //first 5 batches (unscrambled)
      batchedXYpair<FloatType> bxy = batchData(didx.data() + d*batch_size, batch_size, data);
      
      double c1 = model.loss(bxy.x,bxy.y);
      Vector<FloatType> pd = model.deriv();
      
      auto hidden_layer2 = dnn_layer(input_layer<FloatType>(), winit_h, binit_h, ReLU<FloatType>());  
      auto model2 = mse_cost( dnn_layer(hidden_layer2, winit_out, binit_out) );

      std::cout << "Test derivs " << d << " x=" << bxy.x << std::endl;
      for(int i=0;i<p.size(0);i++){
	Vector<FloatType> pp(p);
	doHost(pp, { pp_v(i) += delta; });
	model2.update(pp);
      
	FloatType c2 = model2.loss(bxy.x,bxy.y);
	doHost(pd, {
	    FloatType expect = (c2-c1)/delta;
	    std::cout << i << " got " << pd_v(i) << " expect " << expect << std::endl;
	    if(abs(expect) > 1e-4)
	      assert(abs_near(pd_v(i),expect,FloatType(1e-3)));
	  });
	
      }
    }
  }


  DecayScheduler<FloatType> lr(0.001, 0.1);
  AdamParams<FloatType> ap;
  AdamOptimizer<FloatType, DecayScheduler<FloatType> > opt(ap,lr);
  
  train(model, data, opt, nepoch, batch_size);

  std::cout << "Final params" << std::endl;
  Vector<FloatType> final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    doHost(final_p, { std::cout << i << " " << final_p_v(i) << std::endl; });

  std::cout << "Test on some data" << std::endl;
  FloatType avg_loss= 0.;
  for(int d=0;d<data.size();d++){ 
    auto got = model.predict(data[d].x);
    std::cout << data[d].x << " got " << got << " expect " << data[d].y << std::endl;
    autoView(got_v,got,HostRead);
    autoView(data_v,data[d].y,HostRead);
    avg_loss += pow(got_v(0) - data_v(0),2);
  }
  avg_loss /= data.size();
  std::cout << "Avg. loss " << avg_loss << std::endl;
  assert(avg_loss < 1e-4);
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testOneHiddenLayer();
  
  return 0;
}
