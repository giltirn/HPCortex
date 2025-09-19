#pragma once
#include<MemoryManager.hpp>
#include<Accelerator.hpp>

//An array type whose data movement is controlled by the MemoryManager
//Note, this is not UVM!
template<typename FloatType>
class ManagedArray{
  MemoryManager::HandleIterator handle;
  size_t _size;
 
public:   
  
  ManagedArray(): _size(0){}
  ManagedArray(size_t size, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): _size(size)
  {
    if(size > 0) handle = MemoryManager::globalPool().allocate(size * sizeof(FloatType), alloc_pool);
  }
  ManagedArray(size_t size, FloatType init, MemoryManager::Pool alloc_pool = MemoryManager::Pool::DevicePool): ManagedArray(size, alloc_pool){
    if(size == 0) return;    
    fill(init, alloc_pool);
  }
  ManagedArray(const std::vector<FloatType> &init): ManagedArray(init.size(), MemoryManager::Pool::HostPool){
    if(init.size() == 0) return;
    autoView(vv,(*this),HostWrite);
    memcpy(vv.data(), init.data(), init.size()*sizeof(FloatType));
  }  
  
  ManagedArray(ManagedArray &&r): handle(r.handle), _size(r._size){
    r._size = 0;
  }
  ManagedArray(const ManagedArray &r): _size(0){
    *this = r;
  }

  ManagedArray & operator=(ManagedArray &&r){
    if(_size)
      MemoryManager::globalPool().free(handle);
    _size = r._size;
    handle = r.handle;
    r._size = 0;
    return *this;
  }

  ManagedArray & operator=(const ManagedArray &r){
    if(_size)
      MemoryManager::globalPool().free(handle);
    
    _size = r._size;
    if(_size > 0){
      //We allocate preferentially on the device, but seek to avoid data movement if the device copy is stale      
      if(r.handle->device_in_sync){
	handle = MemoryManager::globalPool().allocate(_size * sizeof(FloatType), MemoryManager::Pool::DevicePool);
	autoView(tv, (*this), DeviceWrite);
	autoView(rv, r, DeviceRead);
	acceleratorCopyDeviceToDevice(tv.data(),rv.data(),_size * sizeof(FloatType));
      }else{
	handle = MemoryManager::globalPool().allocate(_size * sizeof(FloatType), MemoryManager::Pool::HostPool);
	autoView(tv, (*this), HostWrite);
	autoView(rv, r, HostRead);
	memcpy(tv.data(), rv.data(), _size*sizeof(FloatType));
      }
    }
    return *this;
  }
  
  
  inline size_t size() const{ return _size; }
  
  class View{
    FloatType *v;
    size_t _size;
    MemoryManager::HandleIterator handle;
  public:
    accelerator_inline size_t size() const{ return _size; }
    accelerator_inline FloatType* data() const{ return v; }
    accelerator_inline FloatType & operator[](const size_t i) const{ return v[i]; }

    inline View(ViewMode mode, MemoryManager::HandleIterator handle, size_t _size):
      _size(_size), handle(handle), v(_size == 0 ? nullptr : (FloatType*)MemoryManager::globalPool().openView(mode,handle)) {
    }

    inline View(ViewMode mode, const ManagedArray &parent): View(mode, parent.handle, parent._size){}
    
    inline void free(){
      if(_size) MemoryManager::globalPool().closeView(handle);
    }
  };

  inline View view(ViewMode mode) const{
    assert(_size>0);
    return View(mode, *this);
  }
  
  inline ~ManagedArray(){
    if(_size)
      MemoryManager::globalPool().free(handle);
  }

  void fill(FloatType init, MemoryManager::Pool assign_pool = MemoryManager::Pool::DevicePool){
    if(assign_pool == MemoryManager::Pool::DevicePool){
      autoView(hv, (*this), DeviceWrite);
      if(init == FloatType(0.)){
	acceleratorMemSet(hv.data(),0,_size*sizeof(FloatType));
      }else{
	accelerator_for_gen(1,0, splitBlock<32>(), i, _size, {
	    hv[i] = init;
	  });
      }
    }else{
      autoView(hv, (*this), HostWrite);
      thread_for(i, _size, {
	  hv[i] = init;
	});
    }
  }

  inline void lock() const{ MemoryManager::globalPool().lock(handle); }
  inline void unlock() const{ MemoryManager::globalPool().unlock(handle); }

  inline bool deviceResident() const{ return handle->device_in_sync; }
};


/**
 * @brief A container representing an array of managed objects. The associated view allows accessing views of the individual elements by index
 */
template<typename T>
class ManagedTypeArray{
private:
  std::vector<T> elems;
  typedef ManagedArray<typename T::View> ElemViewArray;
  mutable ElemViewArray tv;
  
public:
  ManagedTypeArray(){}
  ManagedTypeArray(int size): elems(size), tv(size, MemoryManager::Pool::HostPool){}

  T & operator[](const int i){ return elems[i]; }
  const T & operator[](const int i) const{ return elems[i]; }
  
  struct View: public ElemViewArray::View{
    ElemViewArray *parent_p;
    
    View(ViewMode mode, ElemViewArray &parent): parent_p(&parent), ElemViewArray::View(mode,parent){}    
    
    void free(){
      {
	autoView(p_v, (*parent_p), HostRead);
	for(int i=0;i<p_v.size();i++) p_v[i].free();
      }
      this->ElemViewArray::View::free();
    }
  };


  View view(ViewMode mode) const{
    //populate views in tv
    {
      autoView(tv_v,tv,HostWrite);
      for(int i=0;i<elems.size();i++)
	tv_v[i] = elems[i].view(mode);
    }
    ViewMode vr_mode;
    switch(mode){
    case DeviceRead:
    case DeviceWrite:
    case DeviceReadWrite:
      vr_mode = DeviceRead; break;
    case HostRead:
    case HostWrite:
    case HostReadWrite:
      vr_mode = HostRead; break;
    default:
      assert(0);
    }
    return View(vr_mode,tv); //open a read view for tv on the appropriate location
  }

  int size() const{ return elems.size(); }

  /**
   * @brief Resize the array, using a lambda function to construct in-place each element. The lambda should take the element index and return the element
   */
  template<typename ElemConstructFunc>
  void resize(int size, const ElemConstructFunc &construct){
    elems.resize(0);
    elems.reserve(size);
    for(int i=0;i<size;i++)
      elems.push_back(construct(i));
    tv = ElemViewArray(size);
  }
  
  void resize(int size){
    elems.resize(size);
    tv = ElemViewArray(size);
  }
    
};
