#include <HPCortex.hpp>
#include <cassert>

void testAccelerator(){
  bool using_omp = false;
#ifdef USE_OMP
  using_omp = true;
#endif

  if(using_omp)   std::cout << "Testing with OpenMP" << std::endl;
  else std::cout << "Testing *without* OpenMP" << std::endl;
  
  //Test threading
  set_threads(2);
  int nthr = thread_max();
  if(using_omp) assert(nthr == 2);
  else assert(nthr == 1);
  
  bool found_1 = false;
  thread_for(i, nthr, {
      int me = thread_num();
      if(me == 1) found_1 = true;
    });

  if(using_omp) assert(found_1 == true);
  else assert(found_1 == false);

  //Set up some inputs and expected outputs of various dimensions
  int n1=5;
  int n2=4;
  int n3=3;
  std::vector<double> in_3d(n1*n2*n3);
  std::vector<double> expect_3d(n1*n2*n3);
  for ( uint64_t i3=0;i3<n3;i3++)
    for ( uint64_t i2=0;i2<n2;i2++)
      for ( uint64_t i1=0;i1<n1;i1++){
	int idx = i1 + n1*(i2 + n2*i3);
	in_3d[idx] = idx;
	expect_3d[idx] = in_3d[idx]*2.13;
      }

  std::vector<double> in_2d(n1*n2);
  std::vector<double> expect_2d(n1*n2);
  for ( uint64_t i2=0;i2<n2;i2++)
    for ( uint64_t i1=0;i1<n1;i1++){
      int idx = i1 + n1*i2;
      in_2d[idx] = idx;
      expect_2d[idx] = in_2d[idx]*2.13;
    }

  std::vector<double> in_1d(n1);
  std::vector<double> expect_1d(n1);
  for ( uint64_t i1=0;i1<n1;i1++){
    int idx = i1;
    in_1d[idx] = idx;
    expect_1d[idx] = in_1d[idx]*2.13;
  }

  
  { //test thread_for3d
    std::vector<double> got(n1*n2*n3);
    thread_for3d(i1,n1,i2,n2,i3,n3,{
	int idx = i1 + n1*(i2 + n2*i3);
	got[idx] = in_3d[idx]*2.13;
      });

    for(int i=0;i<in_3d.size();i++)
      assert(got[i] == expect_3d[i]);
  }
	     
  { //test thread_for2d
    std::vector<double> got(n1*n2);
    thread_for2d(i1,n1,i2,n2,{
	int idx = i1 + n1*i2;
	got[idx] = in_2d[idx]*2.13;
      });

    for(int i=0;i<in_2d.size();i++)
      assert(got[i] == expect_2d[i]);
  }

  { //test thread_for   
    std::vector<double> got(n1);
    thread_for(i1,n1,{
	int idx = i1;
	got[idx] = in_1d[idx]*2.13;
      });

    for(int i=0;i<in_1d.size();i++)
      assert(got[i] == expect_1d[i]);
  }


  { //test accelerator_for3d
    size_t n = n1*n2*n3;
    size_t nB = n*sizeof(double);
    std::vector<double> got(n);
    double* got_device = (double*)acceleratorAllocDevice(nB);
    double* in_device = (double*)acceleratorAllocDevice(nB);
    acceleratorCopyToDevice(in_device, in_3d.data(), nB);
    
    accelerator_for3d(i1,n1,i2,n2,i3,n3,2,{
	int idx = i1 + n1*(i2 + n2*i3);
	got_device[idx] = in_device[idx]*2.13;
      });
    acceleratorCopyFromDevice(got.data(), got_device, nB);
        
    for(int i=0;i<in_3d.size();i++)
      assert(got[i] == expect_3d[i]);
    
      
    acceleratorFreeDevice(got_device);
    acceleratorFreeDevice(in_device);    
  }

 { //test accelerator_for2d
    size_t n = n1*n2;
    size_t nB = n*sizeof(double);
    std::vector<double> got(n);
    double* got_device = (double*)acceleratorAllocDevice(nB);
    double* in_device = (double*)acceleratorAllocDevice(nB);
    acceleratorCopyToDevice(in_device, in_2d.data(), nB);
    
    accelerator_for2d(i1,n1,i2,n2,2,{
	int idx = i1 + n1*i2;
	got_device[idx] = in_device[idx]*2.13;
      });
    acceleratorCopyFromDevice(got.data(), got_device, nB);
        
    for(int i=0;i<in_2d.size();i++)
      assert(got[i] == expect_2d[i]);

    acceleratorFreeDevice(got_device);
    acceleratorFreeDevice(in_device);    
  }

  { //test accelerator_for
    size_t n = n1;
    size_t nB = n*sizeof(double);
    std::vector<double> got(n);
    double* got_device = (double*)acceleratorAllocDevice(nB);
    double* in_device = (double*)acceleratorAllocDevice(nB);
    acceleratorCopyToDevice(in_device, in_1d.data(), nB);
    
    accelerator_for(i1,n1,{
	int idx = i1;
	got_device[idx] = in_device[idx]*2.13;
      });
    acceleratorCopyFromDevice(got.data(), got_device, nB);
        
    for(int i=0;i<in_1d.size();i++)
      assert(got[i] == expect_1d[i]);

    acceleratorFreeDevice(got_device);
    acceleratorFreeDevice(in_device);    
  }

 
  std::cout << "Tests passed" << std::endl;
}


int main(int argc, char** argv){
  initialize(argc,argv);
  
  testAccelerator();

  return 0;
}
