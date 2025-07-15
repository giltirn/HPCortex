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


#endif //USE_CUDA

#ifdef USE_SYCL

template<>
struct SyclItemPos<0>{
  static accelerator_inline int value(sycl::nd_item<3> item){
    return item.get_local_id(2);
  }
};
template<>
struct SyclItemPos<1>{
  static accelerator_inline int value(sycl::nd_item<3> item){
    return item.get_local_id(1);
  }
};
template<>
struct SyclItemPos<2>{
  static accelerator_inline int value(sycl::nd_item<3> item){
    return item.get_local_id(0);	
  }
};
template<>
struct SyclItemPos<3>{
  static accelerator_inline int value(sycl::nd_item<3> item){
    return item.get_group().get_group_id(0);
  }
};
template<>
struct SyclItemPos<4>{
  static accelerator_inline int value(sycl::nd_item<3> item){
    return item.get_group().get_group_id(1);
  }
};
template<>
struct SyclItemPos<5>{
  static accelerator_inline int value(sycl::nd_item<3> item){
    return item.get_group().get_group_id(2);
  }
};

#endif //USE_SYCL
