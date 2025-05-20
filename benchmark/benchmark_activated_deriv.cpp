#include<HPCortex.hpp>
#include<Testing.hpp>

//Compute outer product of incoming derivative from above and the derivative of the activation function
//out(i, b) = above_deriv(i,b) * activation_deriv(i,b)
template<typename FloatType>
Matrix<FloatType> computeThinMatOuterProdBase(const Matrix<FloatType> &above_deriv, const Matrix<FloatType> &activation_deriv){
  int size0 = above_deriv.size(0);
  int batch_size =  above_deriv.size(1);
  assert(activation_deriv.size(0) == size0 && activation_deriv.size(1) == batch_size);
  
  Matrix<FloatType> activated_above_deriv(size0,batch_size);
  autoView(above_deriv_v,above_deriv,DeviceRead);
  autoView(activation_deriv_v,activation_deriv,DeviceRead);
  autoView(activated_above_deriv_v,activated_above_deriv,DeviceWrite);
 
  accelerator_for3d(dummy,1,b,batch_size,i,size0,32,{
      activated_above_deriv_v(i,b) = above_deriv_v(i,b) * activation_deriv_v(i,b);
    });
  return activated_above_deriv;
}

template<typename FloatType>
Matrix<FloatType> computeThinMatOuterProd_v2(const Matrix<FloatType> &above_deriv, const Matrix<FloatType> &activation_deriv){
  int size0 = above_deriv.size(0);
  int batch_size =  above_deriv.size(1);
  assert(activation_deriv.size(0) == size0 && activation_deriv.size(1) == batch_size);
  
  Matrix<FloatType> activated_above_deriv(size0,batch_size);
  autoView(above_deriv_v,above_deriv,DeviceRead);
  autoView(activation_deriv_v,activation_deriv,DeviceRead);
  autoView(activated_above_deriv_v,activated_above_deriv,DeviceWrite);

  int bblocksize = std::min(128,batch_size);
  int nbblocks = (batch_size + bblocksize - 1) / bblocksize; 
  
  accelerator_for3d(bb,bblocksize,bblock,nbblocks,i,size0,1,{      
      int b = bb + bblocksize*bblock;
      if(b < batch_size){      
	activated_above_deriv_v(i,b) = above_deriv_v(i,b) * activation_deriv_v(i,b);
      }
    });
  return activated_above_deriv;
}



int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);
  
  std::vector<int> data_sizes = { 1, 30, 60, 120, 240, 480, 960, 1920 };
  std::vector<int> batch_sizes = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

  for(auto size0 : data_sizes){
    for(auto batch_size : batch_sizes){ 

      Matrix<float> a(size0, batch_size);
      Matrix<float> b(size0, batch_size);
      uniformRandom(a,rng);
      uniformRandom(b,rng);
    
      double mu, sigma;

      Matrix<float> c;
      profileStart();
      benchmark(mu, sigma, 100, 1, [&]{
	c = computeThinMatOuterProd_v2(a,b);
      }, []{});
      profileStop();

      Matrix<float> ctest = computeThinMatOuterProdBase(a,b);
      assert(abs_near(c,ctest,1e-3f,true));
	
	
      std::cout << "size0:" << size0 << "\tbatch: " << batch_size << "\tresult: " << mu/1e-6 << "us " << sigma/1e-6 << "us" << std::endl;
    }      
  }
  return 0;
}

