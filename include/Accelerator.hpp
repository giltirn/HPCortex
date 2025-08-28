#pragma once
#include <HPCortexConfig.h>
#include<strings.h>
#include<cstdlib>
#include<memory.h>
#include<stdio.h>
#include<cassert>
#include<cmath>
#include<iostream>
//Functionality for writing generic GPU kernels with CPU fallback
//Adapted from Peter Boyle's Grid library https://github.com/paboyle/Grid

//- We allow up to 3 dimensions: x,y,z
//- The entire x dimension and a tunable amount of the y direction are iterated over within a block
//- The remainder of the y direction and all of the z direction are iterated over between blocks

void     acceleratorInit(void);
void acceleratorReport();

template<typename decompCoordPolicy, int thrDims, int blockDims, int splitBlockSize>
struct decomp;

#define strong_inline     __attribute__((always_inline)) inline
  
/////////////////////////// CUDA ////////////////////////////////////////////////////////
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_profiler_api.h>
#include "nvtx3/nvToolsExt.h" //TODO: Compile option for nvtx, needs linking to -ldl

#ifdef __CUDA_ARCH__
#define SIMT_ACTIVE
#endif

#define accelerator_only __device__
#define accelerator        __host__ __device__
#define accelerator_inline __host__ __device__ __forceinline__
//inline

extern cudaStream_t copyStream; //stream for async copies
extern cudaStream_t computeStream; //stream for computation

inline void errorCheck(char const *loc, cudaError_t err){
  if(err != cudaSuccess){
    printf("In %s, caught CUDA error: %s\n", loc, cudaGetErrorString( err )); fflush(stdout);
    assert(0);
  }
}

#define accelerator_barrier(dummy)					\
  {									\
    cudaStreamSynchronize(computeStream);				\
    cudaError err = cudaGetLastError();					\
    if ( err != cudaSuccess ) {						\
      printf("accelerator_barrier(): Cuda error %s \n",			\
	     cudaGetErrorString( err ));				\
      printf("File %s Line %d\n",__FILE__,__LINE__);			\
      fflush(stdout);							\
      assert(0);		\
    }									\
  }

inline void *acceleratorAllocHost(size_t bytes)
{
  void *ptr=NULL;
  errorCheck("acceleratorAllocHost", cudaMallocHost((void **)&ptr,bytes));
  return ptr;
}
inline void *acceleratorAllocShared(size_t bytes)
{
  void *ptr=NULL;
  errorCheck("acceleratorAllocShared", cudaMallocManaged((void **)&ptr,bytes));
  return ptr;
};
inline void *acceleratorAllocDevice(size_t bytes)
{
  void *ptr=NULL;
  errorCheck("acceleratorAllocDevice", cudaMalloc((void **)&ptr,bytes));
  return ptr;
};

inline void acceleratorFreeShared(void *ptr){ errorCheck("acceleratorFreeShared", cudaFree(ptr));};
inline void acceleratorFreeDevice(void *ptr){ errorCheck("acceleratorFreeDevice", cudaFree(ptr));};
inline void acceleratorFreeHost(void *ptr){ errorCheck("acceleratorFreeHost", cudaFree(ptr));};
inline void acceleratorCopyToDevice(void* to, void const* from,size_t bytes)  { errorCheck("acceleratorCopyToDevice",cudaMemcpy(to,from,bytes, cudaMemcpyHostToDevice));}
inline void acceleratorCopyFromDevice(void* to, void const* from,size_t bytes){ errorCheck("acceleratorCopyFromDevice", cudaMemcpy(to,from,bytes, cudaMemcpyDeviceToHost));}
inline void acceleratorCopyToDeviceAsync(void* to, void const* from, size_t bytes, cudaStream_t stream = copyStream) { errorCheck("acceleratorCopyToDeviceAsync",cudaMemcpyAsync(to,from,bytes, cudaMemcpyHostToDevice, stream));}
inline void acceleratorCopyFromDeviceAsync(void* to, void const* from, size_t bytes, cudaStream_t stream = copyStream) { errorCheck("acceleratorCopyFromDeviceAsync", cudaMemcpyAsync(to,from,bytes, cudaMemcpyDeviceToHost, stream));}
inline void acceleratorMemSet(void *base,int value,size_t bytes) { errorCheck("acceleratorMemSet",cudaMemset(base,value,bytes));}
inline void acceleratorCopyDeviceToDevice(void* to, void const* from, size_t bytes){
  errorCheck("acceleratorCopyDeviceToDevice",cudaMemcpy(to,from,bytes, cudaMemcpyDeviceToDevice));
}
inline void acceleratorCopyDeviceToDeviceAsynch(void* to, void const* from, size_t bytes) // Asynch
{
  errorCheck("acceleratorCopyDeviceToDeviceAsynch",cudaMemcpyAsync(to,from,bytes, cudaMemcpyDeviceToDevice,copyStream));
}
inline void acceleratorCopySynchronize(void) { errorCheck("acceleratorCopySynchronize",cudaStreamSynchronize(copyStream)); }

accelerator_inline void acceleratorSynchronizeBlock(){
#ifdef SIMT_ACTIVE //workaround
  __syncthreads();
#endif
}

inline void profileStart(){
  cudaProfilerStart();
}
inline void profileStop(){
  cudaProfilerStop();
}
inline void labelRegionBegin(char const* label){
  nvtxRangePush(label);
}
inline void labelRegionEnd(){
  nvtxRangePop();
}

template<int d6>
struct CUDAitemPos;

struct dummyType{};

template<typename lambda>
__global__ void lambdaApply(lambda l){
  extern __shared__ char shared[];
  l(dummyType(),shared);
}

struct decompCoordPolicyCUDA{
  typedef dummyType itemPosContainerType;
  
  template<int Dim>
  static accelerator_inline int itemPos(itemPosContainerType pos){ return CUDAitemPos<Dim>::value(); }
};

#define LAMBDA_MOD mutable
#define DECOMP_POLICY decompCoordPolicyCUDA

template<int thrDim, int blockDim, typename OptionsType, typename Lambda>
void accelerator_for_body(int dims[thrDim+blockDim],
			  const OptionsType &opt,
			  Lambda lambda){
  constexpr int splitBlockSize = OptionsType::splitBlockSize;	\
  decomp<decompCoordPolicyCUDA, thrDim, blockDim, splitBlockSize> decomposition(dims);
  if (decomposition.total_size) {
    dim3 cu_threads(decomposition.decomp_sizes[0],decomposition.decomp_sizes[1],decomposition.decomp_sizes[2]);
    dim3 cu_blocks (decomposition.decomp_sizes[3],decomposition.decomp_sizes[4],decomposition.decomp_sizes[5]);
    lambdaApply<<<cu_blocks,cu_threads,opt.shm_size,computeStream>>>(lambda);
    if(opt.do_barrier) accelerator_barrier(dummy);
  }
}

#define USE_GPU
#endif
//////////////////////////// CUDA ///////////////////////////////////////////////////////


/////////////////////////// HIP ////////////////////////////////////////////////////////
#ifdef USE_HIP
#include <hip/hip_runtime.h>

#ifdef __HIP_DEVICE_COMPILE__
#define SIMT_ACTIVE
#endif

#define accelerator_only   __device__
#define accelerator        __host__ __device__
#define accelerator_inline __host__ __device__ inline

extern hipStream_t copyStream;
extern hipStream_t computeStream;

#define accelerator_barrier(dummy)				\
  {								\
    auto tmp=hipStreamSynchronize(computeStream);		\
    auto err = hipGetLastError();				\
    if ( err != hipSuccess ) {					\
      printf("After hipDeviceSynchronize() : HIP error %s \n", hipGetErrorString( err )); \
      puts(__FILE__);							\
      printf("File %s Line %d\n",__FILE__,__LINE__);			\
      fflush(stdout);						\
      exit(0);							\
    }								\
  }

inline void *acceleratorAllocHost(size_t bytes)
{
  void *ptr=NULL;
  auto err = hipHostMalloc((void **)&ptr,bytes);
  if( err != hipSuccess ) {
    ptr = (void *) NULL;
    fprintf(stderr," hipMallocManaged failed for %ld %s \n",bytes,hipGetErrorString(err)); fflush(stderr);
    assert(0);
  }
  return ptr;
};
inline void *acceleratorAllocShared(size_t bytes)
{
  void *ptr=NULL;
  auto err = hipMallocManaged((void **)&ptr,bytes);
  if( err != hipSuccess ) {
    ptr = (void *) NULL;
    fprintf(stderr," hipMallocManaged failed for %ld %s \n",bytes,hipGetErrorString(err)); fflush(stderr);
    assert(0);
  }
  return ptr;
};

inline void *acceleratorAllocDevice(size_t bytes)
{
  void *ptr=NULL;
  auto err = hipMalloc((void **)&ptr,bytes);
  if( err != hipSuccess ) {
    ptr = (void *) NULL;
    fprintf(stderr," hipMalloc failed for %ld %s \n",bytes,hipGetErrorString(err)); fflush(stderr);
    assert(0);
  }
  return ptr;
};

inline void acceleratorFreeShared(void *ptr){ auto d=hipFree(ptr);};
inline void acceleratorFreeDevice(void *ptr){ auto d=hipFree(ptr);};
inline void acceleratorFreeHost(void *ptr){ auto d=hipFree(ptr);};
inline void acceleratorCopyToDevice(void* to, void const* from,size_t bytes)  { auto d=hipMemcpy(to,from,bytes, hipMemcpyHostToDevice);}
inline void acceleratorCopyFromDevice(void* to, void const* from,size_t bytes){ auto d=hipMemcpy(to,from,bytes, hipMemcpyDeviceToHost);}
inline void acceleratorCopyToDeviceAsync(void* to, void const* from, size_t bytes, hipStream_t stream = copyStream) { auto d=hipMemcpyAsync(to,from,bytes, hipMemcpyHostToDevice, stream);}
inline void acceleratorCopyFromDeviceAsync(void* to, void const* from, size_t bytes, hipStream_t stream = copyStream) { auto d=hipMemcpyAsync(to,from,bytes, hipMemcpyDeviceToHost, stream);}
inline void acceleratorMemSet(void *base,int value,size_t bytes) { auto d=hipMemset(base,value,bytes);}
inline void acceleratorCopyDeviceToDevice(void* to, void const* from, size_t bytes){
  auto d=hipMemcpy(to,from,bytes, hipMemcpyDeviceToDevice);
}
inline void acceleratorCopyDeviceToDeviceAsynch(void* to, void const* from, size_t bytes) // Asynch
{
  auto d=hipMemcpyAsync(to,from,bytes, hipMemcpyDeviceToDevice,copyStream);
}
inline void acceleratorCopySynchronize(void) { auto d=hipStreamSynchronize(copyStream); };

accelerator_inline void acceleratorSynchronizeBlock(){
#ifdef SIMT_ACTIVE //workaround
  __syncthreads();
#endif
}

inline void profileStart(){
}
inline void profileStop(){
}
inline void labelRegionBegin(char const* label){
}
inline void labelRegionEnd(){
}

template<int d6>
struct CUDAitemPos;

struct dummyType{};

template<typename lambda>
__global__ void lambdaApply(lambda l){
  extern __shared__ char shared[];
  l(dummyType(),shared);
}

struct decompCoordPolicyHIP{
  typedef dummyType itemPosContainerType;
  
  template<int Dim>
  static accelerator_inline int itemPos(itemPosContainerType pos){ return CUDAitemPos<Dim>::value(); }
};

#define LAMBDA_MOD mutable
#define DECOMP_POLICY decompCoordPolicyHIP

template<int thrDim, int blockDim, typename OptionsType, typename Lambda>
void accelerator_for_body(int dims[thrDim+blockDim],
			  const OptionsType &opt,
			  Lambda lambda){
  constexpr int splitBlockSize = OptionsType::splitBlockSize;	\
  decomp<decompCoordPolicyHIP, thrDim, blockDim, splitBlockSize> decomposition(dims);
  if (decomposition.total_size) {
    dim3 hip_threads(decomposition.decomp_sizes[0],decomposition.decomp_sizes[1],decomposition.decomp_sizes[2]);
    dim3 hip_blocks (decomposition.decomp_sizes[3],decomposition.decomp_sizes[4],decomposition.decomp_sizes[5]);

    hipLaunchKernelGGL(lambdaApply,hip_blocks,hip_threads, opt.shm_size,computeStream, lambda);		
    if(opt.do_barrier) accelerator_barrier(dummy);
  }
}

#define USE_GPU
#endif
//////////////////////////// HIP ///////////////////////////////////////////////////////


/////////////////////////// SYCL / ONEAPI //////////////////////////////////////////////

#ifdef USE_SYCL

#include <sycl/sycl.hpp>

extern sycl::queue *computeQueue;
extern sycl::queue *copyQueue;

#ifdef __SYCL_DEVICE_ONLY__
#define SIMT_ACTIVE
#endif

#define accelerator_only
#define accelerator 
#define accelerator_inline strong_inline

#define accelerator_barrier(dummy) { computeQueue->wait(); }

inline void *acceleratorAllocShared(size_t bytes){ return malloc_shared(bytes,*computeQueue);};
inline void *acceleratorAllocHost(size_t bytes)  { return malloc_host(bytes,*computeQueue);};
inline void *acceleratorAllocDevice(size_t bytes){ return malloc_device(bytes,*computeQueue);};
inline void acceleratorFreeHost(void *ptr){free(ptr,*computeQueue);};
inline void acceleratorFreeShared(void *ptr){free(ptr,*computeQueue);};
inline void acceleratorFreeDevice(void *ptr){free(ptr,*computeQueue);};

inline void acceleratorCopyToDevice(void* to, const void *from,size_t bytes)  { computeQueue->memcpy(to,from,bytes); computeQueue->wait();}
inline void acceleratorCopyFromDevice(void* to, const void *from,size_t bytes){ computeQueue->memcpy(to,from,bytes); computeQueue->wait();}
inline void acceleratorCopyDeviceToDevice(void* to, const void *from,size_t bytes)  { computeQueue->memcpy(to,from,bytes); computeQueue->wait();}
inline void acceleratorMemSet(void *base,int value,size_t bytes) { computeQueue->memset(base,value,bytes); computeQueue->wait();}

#define acceleratorSynchronizeBlock() pos.barrier(sycl::access::fence_space::local_space)

template<typename FloatType>
inline void atomicAdd(FloatType *p, const FloatType v){
  sycl::atomic_ref<FloatType, sycl::memory_order::relaxed, sycl::memory_scope::device> ap(*p);
  ap.fetch_add(v); 
}

inline void profileStart(){
}
inline void profileStop(){
}
inline void labelRegionBegin(char const* label){
}
inline void labelRegionEnd(){
}

using sycl::min;

template<int d6>
struct SyclItemPos;

struct decompCoordPolicySycl{
  typedef sycl::nd_item<3> itemPosContainerType;
  
  template<int Dim>
  static accelerator_inline int itemPos(itemPosContainerType pos){ return SyclItemPos<Dim>::value(pos); }
};

#define LAMBDA_MOD
//mutable
#define DECOMP_POLICY decompCoordPolicySycl

template<int thrDim, int blockDim, typename OptionsType, typename Lambda>
void accelerator_for_body(int dims[thrDim+blockDim],
			  const OptionsType &opt,
			  Lambda lambda){
  constexpr int splitBlockSize = OptionsType::splitBlockSize;	\
  decomp<decompCoordPolicySycl, thrDim, blockDim, splitBlockSize> decomposition(dims);
  //std::cout << "Sycl split_block_size: " << splitBlockSize << " block_shared_mem: " << opt.shm_size << " blocking: " << opt.do_barrier << std::endl;
  //decomposition.print();
  if (decomposition.total_size) {
    //std::cout << "lcl : " << decomposition.decomp_sizes[2] << " " << decomposition.decomp_sizes[1] << " " << decomposition.decomp_sizes[0] << std::endl;
    //std::cout << "gbl : " << decomposition.decomp_sizes[3]*decomposition.decomp_sizes[2] << " " << decomposition.decomp_sizes[4]*decomposition.decomp_sizes[1] << " " << decomposition.decomp_sizes[5]*decomposition.decomp_sizes[0] << std::endl;			   
    
    computeQueue->submit(
			 [&](sycl::handler &cgh){
			   sycl::range<3> local{
			     size_t(decomposition.decomp_sizes[2]),
			     size_t(decomposition.decomp_sizes[1]),
			     size_t(decomposition.decomp_sizes[0]) //fastest moving dimension is last on sycl
			   };
			   sycl::range<3> global{
			     size_t(decomposition.decomp_sizes[3]*decomposition.decomp_sizes[2]),
			     size_t(decomposition.decomp_sizes[4]*decomposition.decomp_sizes[1]),
			     size_t(decomposition.decomp_sizes[5]*decomposition.decomp_sizes[0])
			   };
			   
			   if(opt.shm_size){			      
			     sycl::local_accessor<char, 1> local_a(sycl::range(opt.shm_size), cgh);
			     
			     cgh.parallel_for(
					      sycl::nd_range<3>(global,local),					    
					      [=] (sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]]{
						char* shared = local_a.get_multi_ptr<sycl::access::decorated::no>().get();
						lambda(item,shared);
					      }
					      );
			   }else{
			     cgh.parallel_for(
					      sycl::nd_range<3>(global,local),					    
					      [=] (sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]]{
						lambda(item,nullptr);
					      }
					      );
			   }
			 }
			 );
    if(opt.do_barrier) accelerator_barrier(dummy);
  }
}

#define USE_GPU
#endif

/////////////////////////// SYCL / ONEAPI //////////////////////////////////////////////



///////////////////////////// OPENMP / NO THREADING //////////////////////////////////////////////////////
//If OMP is detected, use it
#ifdef _OPENMP
#define USE_OMP
#include <omp.h>
#endif

//Host side functionality is always available
#ifdef USE_OMP
#define DO_PRAGMA_(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define thread_num(a) omp_get_thread_num()
#define thread_max(a) omp_get_max_threads()
#define set_threads(a) omp_set_num_threads(a)
#define in_thread_parallel_region(a) omp_in_parallel()
#else
#define DO_PRAGMA_(x) 
#define DO_PRAGMA(x) 
#define thread_num(a) (0)
#define thread_max(a) (1)
#define set_threads(a)
#define in_thread_parallel_region(a) (false)
#endif

#define thread_for( i, num, ... )  \
  DO_PRAGMA(omp parallel for schedule(static)) for ( uint64_t i=0;i<num;i++) { __VA_ARGS__ } ;

#define thread_for3d( i1, n1, i2, n2, i3, n3, ... )	\
  DO_PRAGMA(omp parallel for collapse(3))  \
  for ( uint64_t i3=0;i3<n3;i3++) {	   \
  for ( uint64_t i2=0;i2<n2;i2++) {	   \
  for ( uint64_t i1=0;i1<n1;i1++) {	   \
  { __VA_ARGS__ } ;			   \
  }}}

#define thread_for2d( i1, n1,i2,n2, ... )  \
  DO_PRAGMA(omp parallel for collapse(2))  \
  for ( uint64_t i2=0;i2<n2;i2++) {	   \
  for ( uint64_t i1=0;i1<n1;i1++) {	   \
  { __VA_ARGS__ } ;			   \
  }}


#if !defined(USE_GPU)

#undef SIMT_ACTIVE

#define accelerator
#define accelerator_only
#define accelerator_inline strong_inline

#define accelerator_barrier(dummy) 

inline void acceleratorCopyToDevice(void* to, void const* from,size_t bytes)  { bcopy(from,to,bytes); }
inline void acceleratorCopyFromDevice(void* to, void const* from,size_t bytes){ bcopy(from,to,bytes);}
inline void acceleratorCopyDeviceToDevice(void* to, void const* from,size_t bytes)  { bcopy(from,to,bytes);}
inline void acceleratorCopyDeviceToDeviceAsynch(void* to, void const* from,size_t bytes)  { bcopy(from,to,bytes);}
inline void acceleratorCopySynchronize(void) {};

inline void acceleratorMemSet(void *base,int value,size_t bytes) { memset(base,value,bytes);}

inline void *acceleratorAllocHost(size_t bytes){return malloc(bytes);};
inline void *acceleratorAllocShared(size_t bytes){return malloc(bytes);};
inline void *acceleratorAllocDevice(size_t bytes){return malloc(bytes);};
inline void acceleratorFreeHost(void *ptr){ free(ptr);}
inline void acceleratorFreeShared(void *ptr){free(ptr);};
inline void acceleratorFreeDevice(void *ptr){free(ptr);};

inline void profileStart(){
}
inline void profileStop(){
}
inline void labelRegionBegin(char const* label){
}
inline void labelRegionEnd(){
}

template<typename FloatType>
inline void atomicAdd(FloatType *p, const FloatType v){
#pragma omp atomic
  *p += v;
}

inline void acceleratorSynchronizeBlock(){
#pragma omp barrier
}

using std::min;

struct decompCoordPolicyThread{
  typedef int itemPosContainerType[6];
  
  template<int Dim>
  static accelerator_inline int itemPos(itemPosContainerType pos){ return pos[Dim]; }
};

template<int thrDim, int blockDim, typename OptionsType, typename Lambda>
void accelerator_for_body(int dims[thrDim+blockDim],
		     const OptionsType &opt,
		     Lambda lambda){
  constexpr int splitBlockSize = OptionsType::splitBlockSize;
  decomp<decompCoordPolicyThread, thrDim, blockDim, splitBlockSize> decomposition(dims);

  int nthr = decomposition.decomp_sizes[0]*decomposition.decomp_sizes[1]*decomposition.decomp_sizes[2];
  omp_set_dynamic(0);

  char shared[opt.shm_size];
  
  if (decomposition.total_size) {
    for(int f=0;f<decomposition.decomp_sizes[5];f++){
      for(int e=0;e<decomposition.decomp_sizes[4];e++){
	for(int d=0;d<decomposition.decomp_sizes[3];d++){
#pragma omp parallel num_threads(nthr)
	  {
	    int rem = omp_get_thread_num();
	    int a = rem % decomposition.decomp_sizes[0]; rem /= decomposition.decomp_sizes[0];
	    int b = rem % decomposition.decomp_sizes[1]; rem /= decomposition.decomp_sizes[1];
	    int c = rem;
	    
	    int x[6] = {a,b,c,d,e,f};
	    lambda(x,shared);
	  }
	}
      }
    }
  }
}

//Allow the captured views to be accessed as non-const
#define LAMBDA_MOD mutable
#define DECOMP_POLICY decompCoordPolicyThread

using std::erf;
using std::erff;

#endif // CPU target


////////////////////////////////// GENERAL ///////////////////////////////////////
template<typename decompCoordPolicy, int thrDims, int blockDims, int splitBlockSize>
struct decomp{
  typedef typename decompCoordPolicy::itemPosContainerType itemPosContainerType;
  enum { totalDims = thrDims + blockDims };
  static_assert(thrDims <= 3 && blockDims < 3);
  int decomp_sizes[6];
  int dim_sizes[totalDims]; 
  size_t total_size;
  
  template<int w, bool isThread = (w < thrDims), bool isLastThread = (w == thrDims - 1)>
  struct helper;

  template<int w>
  struct helper<w, true,  false> {
    static accelerator_inline int value(itemPosContainerType pos) { return decompCoordPolicy::template itemPos<w>(pos); }
  };
  template<int w>
  struct helper<w, true,  true> {
    static accelerator_inline int value(itemPosContainerType pos) { return decompCoordPolicy::template itemPos<w>(pos) + splitBlockSize * decompCoordPolicy::template itemPos<3>(pos); }
  }; 
  template<int w, bool na>
  struct helper<w, false, na> {
    static accelerator_inline int value(itemPosContainerType pos) { return decompCoordPolicy::template itemPos<w - thrDims + 4>(pos); }
  };

  template<int w>
  static accelerator_inline int coord(itemPosContainerType pos) {
    return helper<w>::value(pos);
  }
 
  decomp(int _dim_sizes[totalDims]): decomp_sizes{1,1,1,1,1,1}, total_size(1){
    int const* lst = _dim_sizes;
    memcpy(dim_sizes, lst, totalDims*sizeof(int));

    //split the last thread dimension over that and the first dimension of the block
    memcpy(decomp_sizes, lst, thrDims * sizeof(int));
    int nb = (decomp_sizes[thrDims-1] + splitBlockSize - 1) / splitBlockSize;
    decomp_sizes[thrDims-1] = splitBlockSize;
    decomp_sizes[3] = nb;

    memcpy(decomp_sizes+4, lst + thrDims, blockDims * sizeof(int));
    for(int i=0;i<totalDims;i++)
      total_size *= lst[i];
  }
  void print(){
    std::cout << "Blocked decomposition: ";
    for(int i=0;i<6;i++)
      std::cout << decomp_sizes[i] << (i == 2 ? " || " : " ");
    std::cout << std::endl;
  }
  
};

template<typename decompCoordPolicy, int thrDims, int blockDims>
struct decomp<decompCoordPolicy, thrDims,blockDims,1>{
  typedef typename decompCoordPolicy::itemPosContainerType itemPosContainerType;
  enum { totalDims = thrDims + blockDims };
  static_assert(thrDims <= 3 && blockDims <= 3);
  int decomp_sizes[6];
  int dim_sizes[totalDims]; 
  size_t total_size;
  
  template<int w, bool isThread = (w < thrDims)>
  struct helper;

  template<int w>
  struct helper<w, true> {
    static accelerator_inline int value(itemPosContainerType pos){ return decompCoordPolicy::template itemPos<w>(pos);  }
  };
  template<int w>
  struct helper<w, false> {
    static accelerator_inline int value(itemPosContainerType pos){ return decompCoordPolicy::template itemPos<w - thrDims + 3>(pos); }
  };
  template<int w>
  static accelerator_inline int coord(itemPosContainerType pos) {  return helper<w>::value(pos);  }

  decomp(int _dim_sizes[totalDims]): decomp_sizes{1,1,1,1,1,1}, total_size(1){
    int const* lst = _dim_sizes;
    memcpy(dim_sizes, lst, totalDims*sizeof(int));    
    memcpy(decomp_sizes, lst, thrDims * sizeof(int));
    memcpy(decomp_sizes+3, lst + thrDims, blockDims * sizeof(int));
    for(int i=0;i<totalDims;i++)
      total_size *= lst[i];
  }
  void print(){
    std::cout << "Regular decomposition: ";
    for(int i=0;i<6;i++)
      std::cout << decomp_sizes[i] << (i == 2 ? " || " : " ");
    std::cout << std::endl;
  }
};

template<int _splitBlockSize = 1>
struct loopOptions{
  enum { splitBlockSize = _splitBlockSize };
  size_t shm_size;
  bool do_barrier;

  loopOptions(): shm_size(0), do_barrier(true){};

  template<int B>
  inline loopOptions<B> splitBlock(){
    loopOptions<B> out; out.shm_size = shm_size; out.do_barrier = do_barrier; return out;
  }
  inline loopOptions<_splitBlockSize> shm(size_t shm){
    shm_size = shm; return *this;
  }
  inline loopOptions<_splitBlockSize> barrier(bool doit){
    do_barrier = doit; return *this;
  }
  inline loopOptions<_splitBlockSize> normal(){
    return *this;
  } 
};


#define gen_lambda1(iter1, _num1, ...)	\
  int num1 = _num1; \
  int dims[1] = {num1};				\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    if(iter1 < num1){		     \
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_gen(thrDim, blockDim, options, \
			    iter1, num1, ... )	\
  { \
    static_assert(thrDim+blockDim==1);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda1(iter1, num1, { __VA_ARGS__ });		\
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda); \
  };


#define gen_lambda2(iter1, _num1, iter2, _num2, ...)	\
  int num1 = _num1, num2 = _num2;			\
  int dims[2] = {num1,num2};				\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    int iter2 = dimAccessorType::coord<1>(pos); \
    if(iter1 < num1 && iter2 < num2){		     \
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_2d_gen(thrDim, blockDim, options,	\
			      iter1, num1, iter2, num2, ... )	\
  { \
    static_assert(thrDim+blockDim==2);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda2(iter1, num1, iter2, num2, { __VA_ARGS__ });		\
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda); \
  };

#define gen_lambda3(iter1, _num1, iter2, _num2, iter3, _num3, ...) \
  int num1 = _num1, num2 = _num2, num3 = _num3;			\
  int dims[3] = {num1,num2,num3};					\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    int iter2 = dimAccessorType::coord<1>(pos); \
    int iter3 = dimAccessorType::coord<2>(pos);		\
    if(iter1 < num1 && iter2 < num2 && iter3 < num3){			\
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_3d_gen(thrDim, blockDim, options,	\
				iter1, num1, iter2, num2, iter3, num3, ... ) \
  { \
    static_assert(thrDim+blockDim==3);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda3(iter1, num1, iter2, num2, iter3, num3, { __VA_ARGS__ }); \
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda);		\
  };

#define gen_lambda4(iter1, _num1, iter2, _num2, iter3, _num3, iter4, _num4, ...) \
  int num1 = _num1, num2 = _num2, num3 = _num3, num4 = _num4;			\
  int dims[4] = {num1,num2,num3,num4};					\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    int iter2 = dimAccessorType::coord<1>(pos); \
    int iter3 = dimAccessorType::coord<2>(pos);		\
    int iter4 = dimAccessorType::coord<3>(pos);		\
    if(iter1 < num1 && iter2 < num2 && iter3 < num3 && iter4 < num4){			\
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_4d_gen(thrDim, blockDim, options,	\
				iter1, num1, iter2, num2, iter3, num3, iter4, num4, ... ) \
  { \
    static_assert(thrDim+blockDim==4);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda4(iter1, num1, iter2, num2, iter3, num3, iter4, num4, { __VA_ARGS__ }); \
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda);		\
  };

#define gen_lambda5(iter1, _num1, iter2, _num2, iter3, _num3, iter4, _num4, iter5, _num5, ...) \
  int num1 = _num1, num2 = _num2, num3 = _num3, num4 = _num4, num5 = _num5;			\
  int dims[5] = {num1,num2,num3,num4,num5};					\
  auto lambda = [=] accelerator_only (typename dimAccessorType::itemPosContainerType pos, char* shared) LAMBDA_MOD { \
    int iter1 = dimAccessorType::coord<0>(pos); \
    int iter2 = dimAccessorType::coord<1>(pos); \
    int iter3 = dimAccessorType::coord<2>(pos);		\
    int iter4 = dimAccessorType::coord<3>(pos);		\
    int iter5 = dimAccessorType::coord<4>(pos);				\
    if(iter1 < num1 && iter2 < num2 && iter3 < num3 && iter4 < num4 && iter5 < num5){			\
      __VA_ARGS__; \
    }\
  } 

#define accelerator_for_5d_gen(thrDim, blockDim, options,	\
			       iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, ... ) \
  { \
    static_assert(thrDim+blockDim==5);	 \
    auto opt = loopOptions<>().options ; \
    typedef decomp<DECOMP_POLICY, thrDim, blockDim, decltype(opt)::splitBlockSize> dimAccessorType; \
    gen_lambda5(iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, { __VA_ARGS__ }); \
    accelerator_for_body<thrDim,blockDim>(dims, opt, lambda);		\
  };


///////////////////////////// OLD MACROS, UNDERGOING DEPRECATION /////////////////////////////////


#define accelerator_for3dNB( iter1, num1, iter2, num2, iter3, num3, block2, ... ) \
  accelerator_for_3d_gen(1,2,splitBlock<block2>().barrier(false),iter1,num1,iter2,num2,iter3,num3, { __VA_ARGS__ })

#define accelerator_for3dNB_shm( iter1, num1, iter2, num2, iter3, num3, block2, shm_size, ... ) \
  accelerator_for_3d_gen(1,2,splitBlock<block2>().shm(shm_size).barrier(false),iter1,num1,iter2,num2,iter3,num3, { __VA_ARGS__ })

#define accelerator_for_1_3_NB_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, shm_size, ... ) \
  accelerator_for_4d_gen(1,3,splitBlock<block2>().shm(shm_size).barrier(false),iter1,num1,iter2,num2,iter3,num3,iter4,num4, { __VA_ARGS__ })

#define accelerator_for_1_3_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, shm_size, ... ) \
  accelerator_for_4d_gen(1,3,splitBlock<block2>().shm(shm_size),iter1,num1,iter2,num2,iter3,num3,iter4,num4, { __VA_ARGS__ })

#define accelerator_for_1_3_NB( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, ... ) \
  accelerator_for_4d_gen(1,3,splitBlock<block2>().barrier(false),iter1,num1,iter2,num2,iter3,num3,iter4,num4, { __VA_ARGS__ })

#define accelerator_for_1_3( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, ... ) \
  accelerator_for_4d_gen(1,3,splitBlock<block2>(),iter1,num1,iter2,num2,iter3,num3,iter4,num4, { __VA_ARGS__ })

#define accelerator_for_2_3_NB_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, shm_size, ... ) \
  accelerator_for_5d_gen(2,3,shm(shm_size).barrier(false),iter1,num1,iter2,num2,iter3,num3,iter4,num4,iter5,num5, { __VA_ARGS__ })

#define accelerator_for_2_3_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, shm_size, ... ) \
  accelerator_for_5d_gen(2,3,shm(shm_size),iter1,num1,iter2,num2,iter3,num3,iter4,num4,iter5,num5, { __VA_ARGS__ })

#define accelerator_for3d(iter1, num1, iter2, num2, iter3, num3, block2, ... ) \
  accelerator_for3dNB(iter1, num1, iter2, num2, iter3, num3, block2, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_for2dNB(iter1, num1, iter2, num2, block2, ... ) \
accelerator_for3dNB(iter1, num1, iter2, num2, dummy, 1, block2, { __VA_ARGS__ } );

#define accelerator_for2d(iter1, num1, iter2, num2, block2, ... ) \
  accelerator_for2dNB(iter1, num1, iter2, num2, block2, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_forNB( iter1, num1, ... ) accelerator_for3dNB( iter1, num1, dummy1, 1, dummy2,1,  32, {__VA_ARGS__} );  //note iter is over blocks

#define accelerator_for( iter, num, ... )		\
  accelerator_forNB(iter, num, { __VA_ARGS__ } );	\
  accelerator_barrier(dummy);



#define accelerator_for3d_shm(iter1, num1, iter2, num2, iter3, num3, block2, shm_size,... ) \
  accelerator_for3dNB_shm(iter1, num1, iter2, num2, iter3, num3, block2, shm_size,{ __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_for2dNB_shm(iter1, num1, iter2, num2, block2, shm_size, ... ) \
accelerator_for3dNB_shm(iter1, num1, iter2, num2, dummy, 1, block2, shm_size,{ __VA_ARGS__ } );

#define accelerator_for2d_shm(iter1, num1, iter2, num2, block2, shm_size, ... ) \
  accelerator_for2dNB_shm(iter1, num1, iter2, num2, block2, shm_size,{ __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_forNB_shm( iter1, num1, shm_size, ... ) accelerator_for3dNB_shm( iter1, num1, dummy1,1,dummy2,1,  1, shm_size,{__VA_ARGS__} );  

#define accelerator_for_shm( iter, num, shm_size, ... )		\
  accelerator_forNB_shm(iter, num, shm_size,{ __VA_ARGS__ } );	\
  accelerator_barrier(dummy);



//Because View classes cannot have non-trivial destructors, if the view requires a free it needs to be managed externally
//This class calls free on the view. It should be constructed after the view (and be destroyed before, which should happen automatically at the end of the scope)
template<typename ViewType>
struct viewDeallocator{
  ViewType &v;
  viewDeallocator(ViewType &v): v(v){}

  ~viewDeallocator(){
    v.free();
  }

  static void free(ViewType &v){ v.free(); }
};

//Create a view of a managed object and a deallocator that automatically frees it when out of scope
#define autoView(ViewName, ObjName, mode)		\
  auto ViewName = ObjName .view(mode); \
  viewDeallocator<typename std::decay<decltype(ViewName)>::type> ViewName##_d(ViewName);

//Open a HostReadWrite view 'a_v' on managed object 'a' and applies the action
//It is intended to simplify test code
//*NOT INTENDED FOR PERFORMANCE CODE!*
#define doHost(a, ... )\
  {\
    autoView(a##_v,a,HostReadWrite); \
    { __VA_ARGS__ } \
  }
//Same as above for 2 managed objects
#define doHost2(a,b, ... )			\
  {\
    autoView(a##_v,a,HostReadWrite); \
    autoView(b##_v,b,HostReadWrite); \
    { __VA_ARGS__ } \
  }
//3 managed objects...
#define doHost3(a,b,c, ... )			\
  {\
    autoView(a##_v,a,HostReadWrite); \
    autoView(b##_v,b,HostReadWrite); \
    autoView(c##_v,c,HostReadWrite); \
    { __VA_ARGS__ } \
  }
//4 managed objects...
#define doHost4(a,b,c,d, ... )			\
  {\
    autoView(a##_v,a,HostReadWrite); \
    autoView(b##_v,b,HostReadWrite); \
    autoView(c##_v,c,HostReadWrite); \
    autoView(d##_v,d,HostReadWrite); \
    { __VA_ARGS__ } \
  }


#include "implementation/Accelerator.tcc"
