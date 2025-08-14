#include <Accelerator.hpp>
#include <Comms.hpp>
#include <iostream>

#ifdef USE_OMP
#warning "Using OpenMP"
#else
#warning "*NOT* Using OpenMP"
#endif

#if defined(USE_CUDA)
#warning "Compiling with CUDA support"
cudaStream_t copyStream;
cudaStream_t computeStream;

void acceleratorInit(void){
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

#elif defined(USE_HIP)
#warning "Compiling with HIP support"
hipStream_t copyStream;
hipStream_t computeStream;

void acceleratorInit(void){
  int nDevices = 1;
  auto d=hipGetDeviceCount(&nDevices);

  //Distribute node-local ranks evenly over the number of devices round-robin
  //Consider this for optimal GPU-rank binding
  int node_rank = communicators().nodeRank();
  
  int device = node_rank % nDevices;
  
  d=hipSetDevice(device);
  d=hipStreamCreate(&copyStream);
  d=hipStreamCreate(&computeStream);
}

void acceleratorReport(){
  int world_nrank = communicators().worldNrank();
  int world_rank = communicators().worldRank();
  int device;  
  auto d=hipGetDevice(&device);

  int nDevices = 1;
  d=hipGetDeviceCount(&nDevices);
  
  for(int w=0;w<world_nrank;w++){
    assert( MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS );
    if(w == world_rank)
      std::cout << "world:" << world_rank << '/' << world_nrank
	        << " device:" << device << '/' << nDevices
		<< std::endl << std::flush;
  }
}

#elif defined(USE_SYCL)
#warning "Compiling with SYCL support"

sycl::queue *computeQueue;
sycl::queue *copyQueue;

void acceleratorInit(void){
  std::vector<sycl::device> gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int nDevices = gpu_devices.size();
  
  //Distribute node-local ranks evenly over the number of devices round-robin
  //Consider this for optimal GPU-rank binding
  int node_rank = communicators().nodeRank();
  
  int device = node_rank % nDevices;
  
  computeQueue = new sycl::queue (gpu_devices[device]);
  copyQueue = new sycl::queue (gpu_devices[device]);
}
void acceleratorReport(){

}

#else

void  acceleratorInit(void){}
void acceleratorReport(){}

#endif
