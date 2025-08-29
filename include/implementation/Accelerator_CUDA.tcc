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

