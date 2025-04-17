#include <HPCortex.hpp>
#include <Testing.hpp>

//Note, only skipping within a rank, not between!
void testSkipConnectionPipeline(){
  typedef double FloatType;
  
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
    
  int nranks = communicators().pipelineNrank();
  int rank = communicators().pipelineRank();
  
  int batch_size = 1;
  int input_features = 1; 
  int block_output_dims[2] = {1, batch_size};
  int block_input_dims[2] = { rank == nranks-1 ? input_features : 1, batch_size };
  
  FloatType A=3.14;
  FloatType B=0.15;
  FloatType C=5.66;
  FloatType D=-3.455;
  
  Matrix<FloatType> winit1(1,1,A);
  Vector<FloatType> binit1(1,B);
  Matrix<FloatType> winit2(1,1,C);
  Vector<FloatType> binit2(1,D);
  
  if(1){ //test model
    if(!rank) std::cout << "Testing model value pipeline" << std::endl;
    auto skip1 = skip_connection( dnn_layer(input_layer<FloatType>(), winit1,binit1), input_layer<FloatType>());
    auto skip2 = skip_connection( dnn_layer(input_layer<FloatType>(), winit2,binit2), skip1);
        
    auto p = pipeline_block<Matrix<FloatType>, Matrix<FloatType> >( skip2, block_output_dims, block_input_dims);
    
    int value_lag = p.valueLag(); //iterations before first complete cycle of forwards differentiation
    int deriv_lag = p.derivLag(); //iterations before first complete cycle of backwards differentiation
    
    int iters=20;

    //Build the same model on just this rank
    auto test_model = enwrap( input_layer<FloatType>() );
    for(int r=0;r<nranks;r++){
      test_model = enwrap( skip_connection( dnn_layer(input_layer<FloatType>(), winit1,binit1), std::move(test_model) ) ); 
      test_model = enwrap( skip_connection( dnn_layer(input_layer<FloatType>(), winit2,binit2), std::move(test_model) ) ); 
    }
      
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
}


int main(int argc, char** argv){
  initialize(argc, argv);

  testSkipConnectionPipeline();

  return 0;
}

