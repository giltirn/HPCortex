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
	accelerator_for(i, _size, {
	    tv[i] = rv[i];
	  });
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
    accelerator_inline FloatType* data(){ return v; }
    accelerator_inline FloatType const* data() const{ return v; }

    accelerator_inline FloatType & operator[](const size_t i){ return v[i]; }
    accelerator_inline FloatType operator[](const size_t i) const{ return v[i]; }

    inline View(ViewMode mode, MemoryManager::HandleIterator handle, size_t _size):
      _size(_size), handle(handle), v((FloatType*)MemoryManager::globalPool().openView(mode,handle)) {}

    inline View(ViewMode mode, const ManagedArray &parent): View(mode, parent.handle, parent._size){}
    
    inline void free(){
      MemoryManager::globalPool().closeView(handle);
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
      {
	accelerator_for(i, _size, {
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
  
};
