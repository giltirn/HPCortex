#include<HPCortex.hpp>
#include<Testing.hpp>

template<typename FloatType>
Matrix<FloatType> axpyMatThinMatBase(const Matrix<FloatType> &a, const Matrix<FloatType> &b, const Vector<FloatType> &c){
  int size0 = a.size(0);
  assert(c.size(0) == size0);
  int size1 = a.size(1);
  assert(b.size(0) == size1);
  int size2 = b.size(1);
    
  Matrix<FloatType> out(size0,size2);
  {
    autoView(c_v,c,DeviceRead);
    autoView(out_v,out,DeviceWrite);
    autoView(b_v,b,DeviceRead);
    autoView(a_v,a,DeviceRead);

    //Basic version where columns are summed over within a thread and rows/batches distributed over threads
    accelerator_for2d(k,size2,i,size0,1,{
	FloatType v = c_v(i);
	for(int j=0;j<size1;j++)
	  v += a_v(i,j)* b_v(j,k);
	out_v(i,k) = v;	  
      });      
  }
  return out;
}


//O_ik = \sum_i  A_ij B_jk + C_i
template<typename FloatType>
Matrix<FloatType> axpyMatThinMat_v2(const Matrix<FloatType> &a, const Matrix<FloatType> &b, const Vector<FloatType> &c){
#ifdef USE_BLAS
  int sizei = a.size(0);
  int sizej = a.size(1);
  int sizek = b.size(1);

  assert(c.size(0) == sizei);
  assert(b.size(0) == sizej);
  
  Matrix<FloatType> out(sizei,sizek);

  autoView(out_v,out,DeviceWrite);
  autoView(c_v,c,DeviceRead);
 
  accelerator_for_2d_gen(1,1,splitBlock<32>(), k,sizek,i,sizei,{
      out_v(i,k) = c_v(i);
    });
  
  autoView(a_v,a,DeviceRead);
  autoView(b_v,b,DeviceRead);
  
  rmGEMM(NoTranspose,NoTranspose,
	 sizei,sizek,sizej,
	 FloatType(1.0),
	 a_v.data(), sizej,
	 b_v.data(), sizek,
	 FloatType(1.0),
	 out_v.data());
  return out;
#else
  return axpyMatThinMat(a,b,c);
#endif
}

int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);
  
  std::vector<int> data_sizes = { 1, 30, 60, 120, 240, 480, 960, 1920 };
  std::vector<int> batch_sizes = {1, 8, 16, 32, 64};
  
  for(auto data_size : data_sizes){
    for(auto batch_size : batch_sizes){ 

      {
	Matrix<double> a(data_size, data_size);
	Matrix<double> b(data_size, batch_size);
	Vector<double> c(data_size);
	uniformRandom(a,rng);
	uniformRandom(b,rng);
	uniformRandom(c,rng);

	Matrix<double> r = axpyMatThinMat_v2(a,b,c);
	Matrix<double> rtest = axpyMatThinMatBase(a,b,c);
	assert(abs_near(r,rtest,1e-6,true));
      }
      
      Matrix<float> a(data_size, data_size);
      Matrix<float> b(data_size, batch_size);
      Vector<float> c(data_size);
      uniformRandom(a,rng);
      uniformRandom(b,rng);
      uniformRandom(c,rng);
      
      double mu, sigma;

      Matrix<float> r;
      benchmark(mu, sigma, 100, 1, [&]{
	r = axpyMatThinMat_v2(a,b,c);
      }, []{});

      double mu_base, sigma_base;
      benchmark(mu_base, sigma_base, 100, 1, [&]{
	r = axpyMatThinMat(a,b,c);
      }, []{});

      size_t FLOPS = size_t(data_size)*size_t(data_size)*size_t(batch_size)*2 + size_t(data_size);
      
      std::cout << "neurons:" << data_size << "\tbatch: " << batch_size << "\tresult: " << mu/1e-6 << " +/- " << sigma/1e-6 << "us (" << FLOPS/mu/1e9 << " Gflops) base: "
		<< mu_base/1e-6 << " +/- " << sigma_base/1e-6 << "us (" << FLOPS/mu_base/1e9 << " Gflops)" << std::endl;
    }
  }
      
  return 0;
}

