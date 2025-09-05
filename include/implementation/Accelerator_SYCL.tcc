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
inline void acceleratorCopyDeviceToDeviceAsync(void* to, const void *from,size_t bytes)  { copyQueue->memcpy(to,from,bytes); }
inline void acceleratorMemSet(void *base,int value,size_t bytes) { computeQueue->memset(base,value,bytes); computeQueue->wait();}

inline void acceleratorCopySynchronize(void) { copyQueue->wait(); }

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

