#include <hip/hip_runtime.h>

#ifdef __HIP_DEVICE_COMPILE__
#define SIMT_ACTIVE
#endif

//using std::min;

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
