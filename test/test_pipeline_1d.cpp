#include <Layers.hpp>
#include <LossFunctions.hpp>
#include <Pipelining.hpp>
#include <Optimizers.hpp>  
#include <DynamicModel.hpp>

void testPipeline(){
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int batch_size = 1;
  int input_features = 1;
  

  double B=0.15;
  double A=3.14;
  
  Matrix winit(1,1,A);
  Vector binit(1,B);
  typedef decltype( dnn_layer(input_layer(), winit,binit) ) Ltype;


  if(1){ //test model
    if(!rank) std::cout << "Testing model value pipeline" << std::endl;
    //auto b = dnn_layer(input_layer(), winit,binit);    
    //auto p = pipeline_block( b, batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);

    auto p = pipeline_block( dnn_layer(input_layer(), winit,binit), batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);
    
    int value_lag = p.valueLag(); //iterations before first complete cycle of forwards differentiation
    int deriv_lag = p.derivLag(); //iterations before first complete cycle of backwards differentiation
    
    int iters=20;

    //Build the same model on just this rank
    LayerWrapper test_model = enwrap( input_layer() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(std::move(test_model), winit,binit) ); 

    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<Matrix> expect_v(iters);
    std::vector<Vector> expect_d(iters, Vector(test_model.nparams()) );
    
    std::vector<Matrix> input_deriv(iters);
    for(int i=0;i<iters;i++){
      input_deriv[i] = Matrix(1,batch_size, 2.13*(i+1)); 
      Matrix x(1,1, i+1);
      expect_v[i] = test_model.value(x);
      test_model.deriv(expect_d[i],0,input_deriv[i]);
    }
    int nparams = test_model.nparams();

    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      Matrix x(1,1, i+1);
      Matrix v = p.value(x);
      Vector d(nparams,0.);

      int i_vpipe = i-(value_lag-1); //lag=3    2->0  3->1
      int i_dpipe = i-(deriv_lag-1);
      p.deriv(d,i_vpipe >= 0 ? input_deriv[i_vpipe] : Matrix(1,batch_size,-1)); //use the input deriv appropriate to the item index!
      
      if(!rank){

	if(i_vpipe >=0 ){
	  double ev = expect_v[i_vpipe](0,0); 
	  std::cout << i << "\tval expect " << ev << " got "<<  v(0,0) << std::endl;
	}
	if(i_dpipe >=0 ){
	  Vector ed = expect_d[i_dpipe];	
	  std::cout << "\tderiv expect " << ed << " got " << d << std::endl;
	}
      }
    }
  }
  if(1){ //test cost
    if(!rank) std::cout << "Testing loss pipeline" << std::endl;
    auto p = pipeline_block( dnn_layer(input_layer(), winit,binit) , batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);
    PipelineCostFuncWrapper<decltype(p),MSEcostFunc> pc(p);
    int value_lag = p.valueLag();
    int deriv_lag = p.derivLag();
    
    //Build the same model on just this rank
    LayerWrapper test_model = enwrap( input_layer() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(std::move(test_model), winit,binit) ); 
    auto test_cost = mse_cost(test_model);

    int nparams = p.nparams();
    
    int iters=20;

    std::vector<Matrix> x(iters);
    std::vector<Matrix> y(iters);
    
    for(int i=0;i<iters;i++){
      x[i] = Matrix(1,1, i+1);

      double ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      y[i] = Matrix(1,1, 1.05*ival);
    }

    //Get expectation loss and derivatives
    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<double> expect_l(iters);
    std::vector<Vector> expect_d(iters, Vector(test_model.nparams()) );
    for(int i=0;i<iters;i++){
      expect_l[i] = test_cost.loss(x[i],y[i]);
      expect_d[i] = test_cost.deriv();
    }
    
    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      int i_vpipe = i-(value_lag-1);
      double loss = pc.loss(x[i],y[i]);     
      double loss_expect = i_vpipe < 0 ? -1. : expect_l[i_vpipe];

      int i_dpipe = i-(deriv_lag-1); //item index associated with derivative
      Vector deriv = pc.deriv();
      Vector deriv_expect = i_dpipe < 0 ? Vector(nparams,-1.) : expect_d[i_dpipe];
      
      if(!rank){
	std::cout << i << "\tvalue expect " << loss_expect << " got "<<  loss << std::endl;
	std::cout << "\tderiv expect " << deriv_expect << " got " << deriv << std::endl;
      }
    }
  }


  if(1){ //test batched cost
    if(!rank) std::cout << "Testing batch loss pipeline" << std::endl;

    int glob_batch_size = 6*nranks;
    int call_batch_size = 2;
    
    auto p = pipeline_block( dnn_layer(input_layer(), winit,binit) , call_batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);
    BatchPipelineCostFuncWrapper<decltype(p),MSEcostFunc> pc(p, call_batch_size);

    Matrix x(input_features, glob_batch_size);
    Matrix y(1, glob_batch_size);

    for(int i=0;i<glob_batch_size;i++){
      x.pokeColumn(i,Vector(1,i+1));

      double ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      y.pokeColumn(i, Vector(1, 1.05*ival) );
    }
    
    //Build the same model on just this rank
    LayerWrapper test_model = enwrap( input_layer() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(std::move(test_model), winit,binit) ); 
    auto test_cost = mse_cost(test_model);


    double loss_expect = test_cost.loss(x,y);
    Vector deriv_expect = test_cost.deriv();

    double loss_got = pc.loss(x,y);
    Vector deriv_got = pc.deriv();

    if(!rank){
      std::cout << "Loss - got " << loss_got << " expect " << loss_expect << std::endl;
      std::cout << "Deriv - got " << deriv_got << " expect " << deriv_expect << std::endl;
    }
  }
  
}


int main(int argc, char** argv){
  MPI_Init(&argc, &argv);

  testPipeline();

  MPI_Finalize();
  return 0;
}

