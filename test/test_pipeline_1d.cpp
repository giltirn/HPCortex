#include <HPCortex.hpp>
#include <Testing.hpp>

void testPipeline(){
  typedef float FloatType;
  
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
    
  int nranks = communicators().pipelineNrank();
  int rank = communicators().pipelineRank();
  
  int batch_size = 1;
  int input_features = 1; 

  FloatType B=0.15;
  FloatType A=3.14;
  
  Matrix<FloatType> winit(1,1,A);
  Vector<FloatType> binit(1,B);
  typedef decltype( dnn_layer(input_layer<FloatType>(), winit,binit) ) Ltype;


  if(1){ //test model
    if(!rank) std::cout << "Testing model value pipeline" << std::endl;
    //auto b = dnn_layer(input_layer(), winit,binit);    
    //auto p = pipeline_block( b, batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);

    auto p = pipeline_block( dnn_layer(input_layer<FloatType>(), winit,binit), batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);
    
    int value_lag = p.valueLag(); //iterations before first complete cycle of forwards differentiation
    int deriv_lag = p.derivLag(); //iterations before first complete cycle of backwards differentiation
    
    int iters=20;

    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<FloatType>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(std::move(test_model), winit,binit) ); 

    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<Matrix<FloatType> > expect_v(iters);
    std::vector<Vector<FloatType> > expect_d(iters, Vector<FloatType>(test_model.nparams()) );
    
    std::vector<Matrix<FloatType> > input_deriv(iters);
    for(int i=0;i<iters;i++){
      input_deriv[i] = Matrix<FloatType>(1,batch_size, 2.13*(i+1)); 
      Matrix<FloatType> x(1,1, i+1);
      expect_v[i] = test_model.value(x);

      Matrix<FloatType> idcp(input_deriv[i]);
      test_model.deriv(expect_d[i],0,std::move(idcp));
    }
    int nparams = test_model.nparams();

    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      Matrix<FloatType> x(1,1, i+1);
      Matrix<FloatType> v = p.value(x);
      Vector<FloatType> d(nparams,0.);

      int i_vpipe = i-(value_lag-1); //lag=3    2->0  3->1
      int i_dpipe = i-(deriv_lag-1);
      p.deriv(d,i_vpipe >= 0 ? input_deriv[i_vpipe] : Matrix<FloatType>(1,batch_size,-1)); //use the input deriv appropriate to the item index!
      
      if(!rank){

	if(i_vpipe >=0 ){
	  autoView(ev_i_v, expect_v[i_vpipe], HostRead);
	  autoView(v_v,v,HostRead);
	  
	  FloatType ev = ev_i_v(0,0); 
	  std::cout << i << "\tval expect " << ev << " got "<<  v_v(0,0) << std::endl;
	  assert(near(ev,v_v(0,0),FloatType(1e-4)));
	}
	if(i_dpipe >=0 ){
	  Vector<FloatType> ed = expect_d[i_dpipe];	
	  std::cout << "\tderiv expect " << ed << " got " << d << std::endl;
	  assert(near(d,ed,FloatType(1e-4),true));
	}
      }
    }
  }
  if(1){ //test cost
    if(!rank) std::cout << "Testing loss pipeline" << std::endl;
    auto p = pipeline_block( dnn_layer(input_layer<FloatType>(), winit,binit) , batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);
    PipelineCostFuncWrapper<FloatType,decltype(p),MSEcostFunc<Matrix<FloatType>> > pc(p);
    int value_lag = p.valueLag();
    int deriv_lag = p.derivLag();
    
    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<FloatType>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(std::move(test_model), winit,binit) ); 
    auto test_cost = mse_cost(test_model);

    int nparams = p.nparams();
    
    int iters=20;

    std::vector<Matrix<FloatType> > x(iters);
    std::vector<Matrix<FloatType>> y(iters);
    
    for(int i=0;i<iters;i++){
      x[i] = Matrix<FloatType>(1,1, i+1);

      FloatType ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      y[i] = Matrix<FloatType>(1,1, 1.05*ival);
    }

    //Get expectation loss and derivatives
    if(!rank) std::cout << "Computing expectations" << std::endl;
    std::vector<FloatType> expect_l(iters);
    std::vector<Vector<FloatType> > expect_d(iters, Vector<FloatType>(test_model.nparams()) );
    for(int i=0;i<iters;i++){
      expect_l[i] = test_cost.loss(x[i],y[i]);
      expect_d[i] = test_cost.deriv();
    }
    
    if(!rank) std::cout << "Starting test loop" << std::endl;
    for(int i=0;i<iters;i++){
      int i_vpipe = i-(value_lag-1);
      FloatType loss = pc.loss(x[i],y[i]);     
      FloatType loss_expect = i_vpipe < 0 ? -1. : expect_l[i_vpipe];

      int i_dpipe = i-(deriv_lag-1); //item index associated with derivative
      Vector<FloatType> deriv = pc.deriv();
      Vector<FloatType> deriv_expect = i_dpipe < 0 ? Vector<FloatType>(nparams,-1.) : expect_d[i_dpipe];
      
      if(!rank){
	std::cout << i << "\tvalue expect " << loss_expect << " got "<<  loss << std::endl;
	std::cout << "\tderiv expect " << deriv_expect << " got " << deriv << std::endl;
	assert(near(loss_expect,loss,FloatType(1e-4)));
	assert(near(deriv_expect,deriv,FloatType(1e-4),true));
      }
    }
  }


  if(1){ //test batched cost
    if(!rank) std::cout << "Testing batch loss pipeline" << std::endl;

    int glob_batch_size = 6*nranks;
    int call_batch_size = 2;
    
    auto p = pipeline_block( dnn_layer(input_layer<FloatType>(), winit,binit) , call_batch_size, input_features, 1, rank == nranks -1 ? 0 : 1);
    BatchPipelineCostFuncWrapper<FloatType,decltype(p),MSEcostFunc<Matrix<FloatType>> > pc(p, call_batch_size);

    Matrix<FloatType> x(input_features, glob_batch_size);
    Matrix<FloatType> y(1, glob_batch_size);

    for(int i=0;i<glob_batch_size;i++){
      pokeColumn(x,i,Vector<FloatType>(1,i+1));

      FloatType ival = i+1;
      for(int r=0;r<nranks;r++)
	ival = B + A*ival;

      //Add noise
      pokeColumn(y, i, Vector<FloatType>(1, 1.05*ival) );
    }
    
    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<FloatType>() );
    for(int r=0;r<nranks;r++) test_model = enwrap( dnn_layer(std::move(test_model), winit,binit) ); 
    auto test_cost = mse_cost(test_model);


    FloatType loss_expect = test_cost.loss(x,y);
    Vector<FloatType> deriv_expect = test_cost.deriv();

    FloatType loss_got = pc.loss(x,y);
    Vector<FloatType> deriv_got = pc.deriv();

    if(!rank){
      std::cout << "Loss - got " << loss_got << " expect " << loss_expect << std::endl;
      std::cout << "Deriv - got " << deriv_got << " expect " << deriv_expect << std::endl;
      assert(near(loss_expect,loss_got,FloatType(1e-4)));
      assert(near(deriv_expect,deriv_got,FloatType(1e-4),true));
    }
  }
  
}


int main(int argc, char** argv){
  initialize(argc, argv);

  testPipeline();

  return 0;
}

