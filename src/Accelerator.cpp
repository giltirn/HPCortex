#include <Accelerator.hpp>
#include <Comms.hpp>
#include <iostream>

#ifdef USE_OMP
#warning "Using OpenMP"
#else
#warning "*NOT* Using OpenMP"
#endif

#ifdef USE_CUDA
#warning "Compiling with CUDA support"
cudaStream_t copyStream;
cudaStream_t computeStream;
int  acceleratorAbortOnGpuError=1;

void acceleratorInit(void)
{
  int nDevices = 1;
  cudaGetDeviceCount(&nDevices);

  //Distribute node-local ranks evenly over the number of devices round-robin
  //Consider this for optimal GPU-rank binding
  int node_rank = communicators().nodeRank();
  
  int device = node_rank % nDevices;
  
  cudaSetDevice(device);
  cudaStreamCreate(&copyStream);
  cudaStreamCreate(&computeStream);
}

void acceleratorReport(){
  int world_nrank = communicators().worldNrank();
  int world_rank = communicators().worldRank();
  int device;  
  cudaGetDevice(&device);

  int nDevices = 1;
  cudaGetDeviceCount(&nDevices);
  
  for(int w=0;w<world_nrank;w++){
    assert( MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS );
    if(w == world_rank)
      std::cout << "world:" << world_rank << '/' << world_nrank
	        << " device:" << device << '/' << nDevices
		<< std::endl << std::flush;
  }
}


#else

void  acceleratorInit(void){}
void acceleratorReport(){}

#endif
