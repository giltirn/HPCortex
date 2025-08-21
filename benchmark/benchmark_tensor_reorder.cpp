#include <HPCortex.hpp>
#include <Testing.hpp>

#ifdef USE_GPU

template<typename FloatType, int Dim>
void untransformBatchMatrix_base(int rowdim, int coldim, Tensor<FloatType,Dim> &tens, Vector<FloatType> &from){
  assert(rowdim != coldim);
  autoView(tens_v,tens,DeviceWrite);
  autoView(from_v,from,DeviceRead);
  
  accelerator_for_gen(1,0,splitBlock<32>(), i, tens.data_len(),{
      int coord[Dim];
      tensorOffsetUnmap<Dim>(coord, tens_v.sizeArray(), i);

      int o=0;
      for(int d=0;d<Dim;d++) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
      //for(int d=Dim-1;d>=0;d--) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
    
      int off = coord[coldim] + tens_v.size(coldim)*( coord[rowdim] + tens_v.size(rowdim) * o );
      tens_v.data()[i] = from_v(off);
    });
}


void benchmarkUntransform(){
  std::cout << "Benchmarking untransformBatchMatrix\n";
  std::mt19937 rng(1234);
  typedef float FloatType;

  // std::vector<int> osizes = { 1, 5, 8, 16, 32, 33, 64, 128, 256, 512 };
  // std::vector<int> bsizes = { 1, 5, 8, 16, 32, 33, 64 };

  std::vector<int> osizes = { 256 };
  std::vector<int> bsizes = { 64 };
 
  for(auto osize : osizes){
    for(auto bsize : bsizes){
      
      int tsz[3] = {osize,osize,bsize};
      Tensor<float,3> tens(tsz);
  
      Vector<FloatType> arr(tens.data_len());
      uniformRandom(arr, rng);

      for(int rowdim=0;rowdim<2;rowdim++){
	Tensor<float,3> tens_expect(tsz);
	untransformBatchMatrix_base(rowdim, !rowdim, tens_expect, arr);
	untransformBatchMatrix(rowdim, !rowdim, tens, arr);
	assert(equal(tens,tens_expect,true));

	double mu, sigma;
  
	benchmark(mu, sigma, 100, 1, [&]{
	  profileStart();
	  untransformBatchMatrix(rowdim, !rowdim, tens, arr);
	  profileStop();
	}, []{});

	double mu_base, sigma_base;
	benchmark(mu_base, sigma_base, 100, 1, [&]{
	  profileStart();
	  untransformBatchMatrix_base(rowdim, !rowdim, tens_expect, arr);
	  profileStop();
	}, []{});
  
	std::cout << "rowdim:" << rowdim << " sizei:" << tsz[0] << " sizej:" << tsz[1] << " batch_size:" << tsz[2] << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us, base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us" << std::endl;
      }
    }
  }
}

template<typename FloatType, int Dim>
Vector<FloatType> transformBatchMatrix_base(int rowdim, int coldim, const Tensor<FloatType,Dim> &tens){
  assert(rowdim != coldim);
  Vector<FloatType> into(tens.data_len());
  autoView(tens_v,tens,DeviceRead);
  autoView(into_v,into,DeviceWrite);
  
  accelerator_for_gen(1,0,splitBlock<32>(), i, tens.data_len(),{
      int coord[Dim];
      tensorOffsetUnmap<Dim>(coord, tens_v.sizeArray(), i);

      int o=0;
      //for(int d=Dim-1;d>=0;d--) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
      for(int d=0;d<Dim;d++) if(d != rowdim && d != coldim) o = o * tens_v.size(d) + coord[d];
    
      int off = coord[coldim] + tens_v.size(coldim)*( coord[rowdim] + tens_v.size(rowdim) * o );
      into_v(off) = tens_v.data()[i];
    });

  return into;
}

void benchmarkTransform(){
  std::cout << "Benchmarking transformBatchMatrix\n";
  std::mt19937 rng(1234);
  typedef float FloatType;
  
  std::vector<int> osizes = { 1, 5, 8, 16, 32, 33, 64, 128, 256, 512 };
  std::vector<int> bsizes = { 1, 5, 8, 16, 32, 33, 64 };

  //std::vector<int> osizes = { 256 };
  //std::vector<int> bsizes = { 64 };
 
  for(auto osize : osizes){
    for(auto bsize : bsizes){
      int tsz[3] = {osize,osize,bsize};
      Tensor<float,3> tens(tsz);
      uniformRandom(tens, rng);
      
      for(int rowdim=0;rowdim<2;rowdim++){
	Tensor<float,3> tens_expect(tsz);
	Vector<FloatType> expect = transformBatchMatrix_base(rowdim, !rowdim, tens);
	Vector<FloatType> got = transformBatchMatrix(rowdim, !rowdim, tens);
	assert(equal(got,expect,true));

	double mu, sigma;
  
	benchmark(mu, sigma, 100, 1, [&]{
	  profileStart();
	  got = transformBatchMatrix(rowdim, !rowdim, tens);
	  profileStop();
	}, []{});

	double mu_base, sigma_base;
	benchmark(mu_base, sigma_base, 100, 1, [&]{
	  profileStart();
	  got = transformBatchMatrix_base(rowdim, !rowdim, tens);
	  profileStop();
	}, []{});
  
	std::cout << "rowdim:" << rowdim << " sizei:" << tsz[0] << " sizej:" << tsz[1] << " batch_size:" << tsz[2] << "\tresult: " << mu/1e-6 << "+-" << sigma/1e-6 << "us, base: " << mu_base/1e-6 << "+-" << sigma_base/1e-6 << "us" << std::endl;
      }
    }
  }


}

int main(int argc, char** argv){  
  initialize(argc,argv);
  benchmarkUntransform();
  benchmarkTransform();
  
  return 0;
}

#else
int main(void){
  return 0;
}
#endif
