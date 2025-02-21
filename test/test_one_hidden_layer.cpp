#include <Layers.hpp>
#include <LossFunctions.hpp>
#include <Optimizers.hpp>  

void testOneHiddenLayer(){
  //Test f(x) = 0.2*x + 0.3;
  int nbatch = 100;
  int batch_size = 4;
  std::vector<XYpair> data(nbatch);

  int ndata = batch_size * nbatch;

  for(int i=0;i<ndata;i++){ //i = b + batch_size * B
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1

    int b = i % batch_size;
    int B = i / batch_size;
    if(b==0){
      data[B].x = Matrix(1,batch_size);
      data[B].y = Matrix(1,batch_size);
    }
    
    data[B].x(0,b) = x;
    data[B].y(0,b) = 0.2*x + 0.3;
  }

  int nhidden = 5;

  Matrix winit_out(1,nhidden,0.01);
  Matrix winit_h(nhidden,1,0.01);

  Vector binit_out(1,0.01);
  Vector binit_h(nhidden, 0.01);

  auto hidden_layer( dnn_layer(input_layer(), winit_h, binit_h, ReLU()) );
  auto model = mse_cost( dnn_layer(hidden_layer, winit_out, binit_out) );

  //Test derivative
  {
    Vector p = model.getParams();
    
    for(int d=1;d<5;d++){ //first 5 data
    
      double c1 = model.loss(data[d].x,data[d].y);
      Vector pd = model.deriv();
      
      auto hidden_layer2 = dnn_layer(input_layer(), winit_h, binit_h, ReLU());  
      auto model2 = mse_cost( dnn_layer(hidden_layer2, winit_out, binit_out) );

      std::cout << "Test derivs " << d << " x=" << data[d].x(0,0) << " " << data[d].x(0,1) << std::endl;
      for(int i=0;i<p.size(0);i++){
	Vector pp(p);
	pp(i) += 1e-9;
	model2.update(pp);
      
	double c2 = model2.loss(data[d].x,data[d].y);
	std::cout << i << " got " << pd(i) << " expect " << (c2-c1)/1e-9 << std::endl;
      }
    }
  }


  DecayScheduler lr(0.001, 0.1);
  AdamParams ap;
  optimizeAdam(model, data, lr, ap, 200);

  std::cout << "Final params" << std::endl;
  Vector final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p(i) << std::endl;

  std::cout << "Test on some data" << std::endl;
  for(int d=0;d<data.size();d++){ //first 5 data, batch idx 0
    auto got = model.predict(data[d].x);
    std::cout << data[d].x(0,0) << " got " << got(0,0) << " expect " << data[d].y(0,0) << std::endl;
  }

}

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  
  testOneHiddenLayer();
  
  MPI_Finalize();
  return 0;
}
