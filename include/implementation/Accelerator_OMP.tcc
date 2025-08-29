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
