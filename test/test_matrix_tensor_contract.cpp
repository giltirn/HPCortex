#include <HPCortex.hpp>
#include <Testing.hpp>

void testMatrixTensorContract(){
  typedef double FloatType;
  std::mt19937 rng(1234);
   
  typedef std::vector<FloatType> vecD;

  int size0 = 5;
  int size1 = 3;
  int batch_size = 5;
  Matrix<FloatType> winit(size0,size1);
  uniformRandom(winit, rng);

  typedef Tensor<FloatType,4> Tens;
  
  int in_sizes[4] = {2,3,size1,batch_size};
  int out_sizes[4] = {2,3,size0,batch_size};
  
  auto m = matrix_tensor_contract_layer<4>(winit, input_layer<FloatType, Tens>());

  Tens in(in_sizes);
  uniformRandom(in, rng);

  Tens vgot = m.value(in);
  Tens vexpect(out_sizes,0.0);
  doHost3(vexpect, winit, in, {  
      for(int i=0;i<2;i++){
	for(int j=0;j<3;j++){
	  for(int k=0; k<size0;k++){
	    for(int b=0; b<batch_size; b++){
	      
	      for(int kk=0; kk< size1; kk++)
		vexpect_v(i,j,k,b) += winit_v(k,kk) * in_v(i,j,kk,b);
	    }
	  }
	}
      }
    });
	 
  assert(abs_near(vgot,vexpect,FloatType(1e-4),true));
  
  assert(m.nparams() == size0*size1);

  testDeriv(m, in_sizes, out_sizes);

  std::cout << "Tests passed" << std::endl;
}

void testbatch3tensorContract(){
  typedef double FloatType;
  std::mt19937 rng(1234);
   
  typedef std::vector<FloatType> vecD;

  //0 with 0
  {
    std::cout << "Test 0,0" << std::endl;      
    int sizeA[3] = {3,5,4};
    int sizeB[3] = {3,6,4};
    
    Tensor<FloatType,3> A(sizeA), B(sizeB);
    uniformRandom(A,rng);
    uniformRandom(B,rng);
    Tensor<FloatType,3> C = batch3tensorContract(A,B,0,0);
    assert(C.size(0) == 5 && C.size(1) == 6 && C.size(2) == 4);

    int sizeC[3] = {5,6,4};
    Tensor<FloatType,3> Cexpect(sizeC,0.);
    doHost3(A,B,Cexpect, {
	for(int i=0;i<5;i++){
	  for(int j=0;j<6;j++){
	    for(int b=0;b<4;b++){

	      for(int k=0;k<3;k++)
		Cexpect_v(i,j,b) += A_v(k,i,b)*B_v(k,j,b);
	    }
	  }
	}
      });
    assert(abs_near(Cexpect,C,1e-5,true));
  }
  //0 with 1
  {
    std::cout << "Test 0,1" << std::endl;
    int sizeA[3] = {3,5,4};
    int sizeB[3] = {7,3,4};
    
    Tensor<FloatType,3> A(sizeA), B(sizeB);
    uniformRandom(A,rng);
    uniformRandom(B,rng);
    Tensor<FloatType,3> C = batch3tensorContract(A,B,0,1);
    assert(C.size(0) == 5 && C.size(1) == 7 && C.size(2) == 4);

    int sizeC[3] = {5,7,4};
    Tensor<FloatType,3> Cexpect(sizeC,0.);
    doHost3(A,B,Cexpect, {
	for(int i=0;i<5;i++){
	  for(int j=0;j<7;j++){
	    for(int b=0;b<4;b++){

	      for(int k=0;k<3;k++)
		Cexpect_v(i,j,b) += A_v(k,i,b)*B_v(j,k,b);
	    }
	  }
	}
      });
    assert(abs_near(Cexpect,C,1e-5,true));
  }
  //1 with 0
  {
    std::cout << "Test 1,0" << std::endl;
    int sizeA[3] = {3,5,4};
    int sizeB[3] = {5,4,4};
    
    Tensor<FloatType,3> A(sizeA), B(sizeB);
    uniformRandom(A,rng);
    uniformRandom(B,rng);
    Tensor<FloatType,3> C = batch3tensorContract(A,B,1,0);
    assert(C.size(0) == 3 && C.size(1) == 4 && C.size(2) == 4);

    int sizeC[3] = {3,4,4};
    Tensor<FloatType,3> Cexpect(sizeC,0.);
    doHost3(A,B,Cexpect, {
	for(int i=0;i<3;i++){
	  for(int j=0;j<4;j++){
	    for(int b=0;b<4;b++){

	      for(int k=0;k<5;k++)
		Cexpect_v(i,j,b) += A_v(i,k,b)*B_v(k,j,b);
	    }
	  }
	}
      });
    assert(abs_near(Cexpect,C,1e-5,true));
  }
  //1 with 1
  {
    std::cout << "Test 1,1" << std::endl;
    int sizeA[3] = {3,5,4};
    int sizeB[3] = {6,5,4};
    
    Tensor<FloatType,3> A(sizeA), B(sizeB);
    uniformRandom(A,rng);
    uniformRandom(B,rng);
    Tensor<FloatType,3> C = batch3tensorContract(A,B,1,1);
    assert(C.size(0) == 3 && C.size(1) == 6 && C.size(2) == 4);

    int sizeC[3] = {3,6,4};
    Tensor<FloatType,3> Cexpect(sizeC,0.);
    doHost3(A,B,Cexpect, {
	for(int i=0;i<3;i++){
	  for(int j=0;j<6;j++){
	    for(int b=0;b<4;b++){

	      for(int k=0;k<5;k++)
		Cexpect_v(i,j,b) += A_v(i,k,b)*B_v(j,k,b);
	    }
	  }
	}
      });
    assert(abs_near(Cexpect,C,1e-5,true));
  }





  
}



  

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testMatrixTensorContract();
  testbatch3tensorContract();
  return 0;
}
