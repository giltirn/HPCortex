#ifdef USE_CUDA

template<>
struct CUDAitemPos<0>{
  static accelerator_inline int value(){
#ifdef SIMT_ACTIVE
    return threadIdx.x;
#else
    return 0;
#endif
  }
};
template<>
struct CUDAitemPos<1>{
  static accelerator_inline int value(){
#ifdef SIMT_ACTIVE
    return threadIdx.y;
#else
    return 0;
#endif
  }
};
template<>
struct CUDAitemPos<2>{
  static accelerator_inline int value(){
#ifdef SIMT_ACTIVE
    return threadIdx.z;
#else
    return 0;
#endif
  }
};
template<>
struct CUDAitemPos<3>{
  static accelerator_inline int value(){
#ifdef SIMT_ACTIVE
    return blockIdx.x;
#else
    return 0;
#endif
  }
};
template<>
struct CUDAitemPos<4>{
  static accelerator_inline int value(){
#ifdef SIMT_ACTIVE
    return blockIdx.y;
#else
    return 0;
#endif
  }
};
template<>
struct CUDAitemPos<5>{
  static accelerator_inline int value(){
#ifdef SIMT_ACTIVE
    return blockIdx.z;
#else
    return 0;
#endif
  }
};



#endif
