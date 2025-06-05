#pragma once
#include <HPCortexConfig.h>
#include<strings.h>
#include<cstdlib>
#include<memory.h>
#include<stdio.h>
#include<cassert>
//Functionality for writing generic GPU kernels with CPU fallback
//Adapted from Peter Boyle's Grid library https://github.com/paboyle/Grid

//- We allow up to 3 dimensions: x,y,z
//- The entire x dimension and a tunable amount of the y direction are iterated over within a block
//- The remainder of the y direction and all of the z direction are iterated over between blocks


void     acceleratorInit(void);
void acceleratorReport();

/////////////////////////// CUDA ////////////////////////////////////////////////////////
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_profiler_api.h>
#include "nvtx3/nvToolsExt.h" //TODO: Compile option for nvtx, needs linking to -ldl

#ifdef __CUDA_ARCH__
#define SIMT_ACTIVE
#endif

#define accelerator        __host__ __device__
#define accelerator_inline __host__ __device__ __forceinline__
//inline

extern int acceleratorAbortOnGpuError;
extern cudaStream_t copyStream; //stream for async copies
extern cudaStream_t computeStream; //stream for computation

//Baseline call allows for up to 3 dimensions
template<typename lambda>  __global__
void LambdaApply(uint64_t num1, uint64_t num2, uint64_t num3, uint64_t block2, lambda Lambda)
{
  uint64_t x = threadIdx.x; //all of num1 within the block
  uint64_t y = threadIdx.y + block2*blockIdx.x; //note ordering of cu_blocks indices below
  uint64_t z = blockIdx.y;
  
  if ( (x < num1) && (y<num2) && (z<num3) ) {
    Lambda(x,y,z);
  }
}

//block2 is the number of y elements to iterate over within a block
//ideally should be a divisor of num2 but not required
#define accelerator_for3dNB( iter1, num1, iter2, num2, iter3, num3, block2, ... ) \
  {                                                                     \
    if ( num1*num2*num3 ) {							\
      typedef uint64_t Iterator;					\
      auto lambda = [=] accelerator					\
	(Iterator iter1,Iterator iter2,Iterator iter3) mutable {	\
		      __VA_ARGS__;					\
		    };							\
      dim3 cu_threads(num1,block2,1);			\
      dim3 cu_blocks ((num2+block2-1)/block2,num3,1);				\
      LambdaApply<<<cu_blocks,cu_threads,0,computeStream>>>(num1,num2,num3,block2,lambda); \
    }									\
  }

#define accelerator_for3dNB_shm( iter1, num1, iter2, num2, iter3, num3, block2, shm_size, ... ) \
  {									\
    if ( num1*num2*num3 ) {							\
      typedef uint64_t Iterator;					\
      auto lambda = [=] __device__					\
	(Iterator iter1,Iterator iter2,Iterator iter3) mutable {	\
		      __VA_ARGS__;					\
		    };							\
      dim3 cu_threads(num1,block2,1);			\
      dim3 cu_blocks ((num2+block2-1)/block2,num3,1);				\
      LambdaApply<<<cu_blocks,cu_threads,shm_size,computeStream>>>(num1,num2,num3,block2,lambda); \
    }									\
  }

template<typename lambda>  __global__
void LambdaApply1_3(uint64_t num1, uint64_t num2, uint64_t num3, uint64_t num4, uint64_t block2, lambda Lambda)
{
  uint64_t x = threadIdx.x; //all of num1 within the block
  uint64_t y = threadIdx.y + block2*blockIdx.x; //note ordering of cu_blocks indices below
  uint64_t z = blockIdx.y;
  uint64_t t = blockIdx.z;
  
  if ( (x < num1) && (y<num2) && (z<num3) && (t<num4) ) {
    Lambda(x,y,z,t);
  }
}

//1 iterable within block, 3 over blocks
#define accelerator_for_1_3_NB_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, shm_size, ... ) \
  {									\
    if ( num1*num2*num3*num4 ) {							\
      typedef uint64_t Iterator;					\
      auto lambda = [=] __device__					\
	(Iterator iter1,Iterator iter2,Iterator iter3, Iterator iter4) mutable {	\
		      __VA_ARGS__;					\
		    };							\
      dim3 cu_threads(num1,block2,1);			\
      dim3 cu_blocks ((num2+block2-1)/block2,num3,num4);				\
      LambdaApply1_3<<<cu_blocks,cu_threads,shm_size,computeStream>>>(num1,num2,num3,num4,block2,lambda); \
    }									\
  }

#define accelerator_for_1_3_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, block2, shm_size, ... ) \
  accelerator_for_1_3_NB_shm(iter1,num1,iter2,num2,iter3,num3,iter4,num4,block2,shm_size,{__VA_ARGS__}) \
  accelerator_barrier(dummy)



template<typename lambda>  __global__
void LambdaApply2_3(uint64_t num1, uint64_t num2, uint64_t num3, uint64_t num4, uint64_t num5, lambda Lambda)
{
  uint64_t x = threadIdx.x;
  uint64_t y = threadIdx.y;  
  uint64_t z = blockIdx.x;
  uint64_t t = blockIdx.y;
  uint64_t u = blockIdx.z;
  
  if ( (x < num1) && (y<num2) && (z<num3) && (t<num4) && (u<num5) ){
    Lambda(x,y,z,t,u);
  }
}

//2 iterable within block, 3 over blocks
#define accelerator_for_2_3_NB_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, shm_size, ... ) \
  {									\
    if ( num1*num2*num3*num4*num5 ) {							\
      typedef uint64_t Iterator;					\
      auto lambda = [=] __device__					\
	(Iterator iter1,Iterator iter2,Iterator iter3, Iterator iter4, Iterator iter5) mutable { \
		      __VA_ARGS__;					\
		    };							\
      dim3 cu_threads(num1,num2,1);			\
      dim3 cu_blocks (num3,num4,num5);					\
      LambdaApply2_3<<<cu_blocks,cu_threads,shm_size,computeStream>>>(num1,num2,num3,num4,num5,lambda); \
    }									\
  }

#define accelerator_for_2_3_shm( iter1, num1, iter2, num2, iter3, num3, iter4, num4, iter5, num5, shm_size, ... ) \
  accelerator_for_2_3_NB_shm(iter1,num1,iter2,num2,iter3,num3,iter4,num4,iter5,num5,shm_size,{__VA_ARGS__}) \
  accelerator_barrier(dummy)





#define accelerator_barrier(dummy)					\
  {									\
    cudaStreamSynchronize(computeStream);				\
    cudaError err = cudaGetLastError();					\
    if ( cudaSuccess != err ) {						\
      printf("accelerator_barrier(): Cuda error %s \n",			\
	     cudaGetErrorString( err ));				\
      printf("File %s Line %d\n",__FILE__,__LINE__);			\
      fflush(stdout);							\
      if (acceleratorAbortOnGpuError) assert(err==cudaSuccess);		\
    }									\
  }

inline void *acceleratorAllocHost(size_t bytes)
{
  void *ptr=NULL;
  auto err = cudaMallocHost((void **)&ptr,bytes);
  if( err != cudaSuccess ) {
    ptr = (void *) NULL;
    printf(" cudaMallocHost failed for %lu %s \n",bytes,cudaGetErrorString(err));
    assert(0);
  }
  return ptr;
}
inline void *acceleratorAllocShared(size_t bytes)
{
  void *ptr=NULL;
  auto err = cudaMallocManaged((void **)&ptr,bytes);
  if( err != cudaSuccess ) {
    ptr = (void *) NULL;
    printf(" cudaMallocManaged failed for %lu %s \n",bytes,cudaGetErrorString(err));
    assert(0);
  }
  return ptr;
};
inline void *acceleratorAllocDevice(size_t bytes)
{
  void *ptr=NULL;
  auto err = cudaMalloc((void **)&ptr,bytes);
  if( err != cudaSuccess ) {
    ptr = (void *) NULL;
    printf(" cudaMalloc failed for %lu %s \n",bytes,cudaGetErrorString(err));
  }
  return ptr;
};

inline void acceleratorFreeShared(void *ptr){ cudaFree(ptr);};
inline void acceleratorFreeDevice(void *ptr){ cudaFree(ptr);};
inline void acceleratorFreeHost(void *ptr){ cudaFree(ptr);};
inline void acceleratorCopyToDevice(void* to, void const* from,size_t bytes)  { cudaMemcpy(to,from,bytes, cudaMemcpyHostToDevice);}
inline void acceleratorCopyFromDevice(void* to, void const* from,size_t bytes){ cudaMemcpy(to,from,bytes, cudaMemcpyDeviceToHost);}
inline void acceleratorCopyToDeviceAsync(void* to, void const* from, size_t bytes, cudaStream_t stream = copyStream) { cudaMemcpyAsync(to,from,bytes, cudaMemcpyHostToDevice, stream);}
inline void acceleratorCopyFromDeviceAsync(void* to, void const* from, size_t bytes, cudaStream_t stream = copyStream) { cudaMemcpyAsync(to,from,bytes, cudaMemcpyDeviceToHost, stream);}
inline void acceleratorMemSet(void *base,int value,size_t bytes) { cudaMemset(base,value,bytes);}
inline void acceleratorCopyDeviceToDevice(void* to, void const* from, size_t bytes){
  cudaMemcpy(to,from,bytes, cudaMemcpyDeviceToDevice);
}
inline void acceleratorCopyDeviceToDeviceAsynch(void* to, void const* from, size_t bytes) // Asynch
{
  cudaMemcpyAsync(to,from,bytes, cudaMemcpyDeviceToDevice,copyStream);
}
inline void acceleratorCopySynchronize(void) { cudaStreamSynchronize(copyStream); };

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

#endif
//////////////////////////// CUDA ///////////////////////////////////////////////////////






///////////////////////////// OPENMP / NO THREADING //////////////////////////////////////////////////////
//If OMP is detected, use it
#ifdef _OPENMP
#define USE_OMP
#include <omp.h>
#endif

#define strong_inline     __attribute__((always_inline)) inline

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


#if !defined(USE_CUDA)

#undef SIMT_ACTIVE

#define accelerator 
#define accelerator_inline strong_inline

#define accelerator_barrier(dummy) 
#define accelerator_for3dNB( iter1, num1, iter2, num2, iter3, num3, block2, ... ) thread_for3d( iter1, num1, iter2, num2, iter3, num3, { __VA_ARGS__ } )


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

#endif // CPU target


////////////////////////////////// GENERAL ///////////////////////////////////////
#define accelerator_for3d(iter1, num1, iter2, num2, iter3, num3, block2, ... ) \
  accelerator_for3dNB(iter1, num1, iter2, num2, iter3, num3, block2, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

#define accelerator_for2dNB(iter1, num1, iter2, num2, block2, ... ) \
accelerator_for3dNB(iter1, num1, iter2, num2, dummy, 1, block2, { __VA_ARGS__ } );

#define accelerator_for2d(iter1, num1, iter2, num2, block2, ... ) \
  accelerator_for2dNB(iter1, num1, iter2, num2, block2, { __VA_ARGS__ } ); \
  accelerator_barrier(dummy);

//#define accelerator_forNB( iter1, num1, ... ) accelerator_for3dNB( dummy1,1, iter1, num1, dummy2,1,  1, {__VA_ARGS__} );  //note iter is over blocks
#define accelerator_forNB( iter1, num1, ... ) accelerator_for3dNB( dummy1,1, iter1, num1, dummy2,1,  32, {__VA_ARGS__} );  //note iter is over blocks

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
