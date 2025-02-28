#include<MemoryManager.hpp>
#include<Comms.hpp>
#include<Accelerator.hpp>
#include<Timing.hpp>
#include<cstdio>
#include<fstream>
#include<sstream>
#include <sys/mman.h>

static inline void* mmap_alloc(const size_t byte_size){
  void *p = mmap (NULL, byte_size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, (off_t)0);
  if(p == MAP_FAILED){
    std::ostringstream os; os << "Failed to mmap an anonymous region of size " << byte_size << "B";
    throw std::runtime_error(os.str());
  }
  return p;
}
static inline void mmap_free(void* pv, const size_t byte_size){
  if( munmap(pv, byte_size) != 0){
    std::ostringstream os; os << "Failed to munmap an anonymous region of size " << byte_size << "B";
    throw std::runtime_error(os.str());
  }
}

static inline double byte_to_MB(size_t B){
  return double(B)/1024./1024.;
}


void MemoryManager::enableIOlogging(){
  this->io_logger = new std::ofstream("mempool_io." + std::to_string(UniqueID()) + ".log", std::ofstream::out | std::ofstream::trunc );
}

MemoryManager::EntryIterator MemoryManager::allocEntry(size_t bytes, Pool pool){
  sanityCheck();
  Entry e;
  e.bytes = bytes;
  e.owned_by = nullptr;
  if(pool == DevicePool){
    e.ptr = acceleratorAllocDevice(bytes);
    device_allocated += bytes;
    device_allocated_HWM = std::max(device_allocated_HWM, device_allocated);
    if(verbose) std::cout << "MemoryManager: Allocated device entry " << e.ptr << " of size " << bytes << ". Allocated amount is now " << device_allocated << " vs max " << device_pool_max_size << std::endl;
  }else{ //HostPool
    e.ptr = mmap_alloc(bytes);
    host_allocated += bytes;
    host_allocated_HWM = std::max(host_allocated_HWM, host_allocated);
    if(verbose) std::cout << "MemoryManager: Allocated host entry " << e.ptr << " of size " << bytes << ". Allocated amount is now " << host_allocated << " vs max " << host_pool_max_size << std::endl;
  }
  auto &p = getLRUpool(pool);
  return p.insert(p.end(),e);
}

void MemoryManager::sanityCheck(){
#ifdef ENABLE_MEMMAN_SANITY_CHECK
  for(int p=0;p<2;p++){
    Pool pool = p==0 ? DevicePool : HostPool;
    auto const &fp = getFreePool(pool);
    for(auto fit = fp.begin(); fit != fp.end(); fit++){
      auto const &entry_list = fit->second;
      size_t bytes = fit->first;
      size_t sz = entry_list.size();
      size_t cnt =0;
      for(auto it = entry_list.begin(); it != entry_list.end(); it++){	  
	++cnt;
	if(it->bytes != bytes){
	  std::ostringstream os; 
	  os << "Found entry of size " << it->bytes << " in " << poolName(pool) << " free pool for size " << bytes << "!";
	  throw std::runtime_error(os.str());
	}
      }
      if(cnt != sz){
	std::ostringstream os; 
	os << "Mismatch in list size: list claims " << sz << " but counted " << cnt << ". Corruption?";
	throw std::runtime_error(os.str());
      }
    }
  }
#endif
}

void MemoryManager::moveEntryToFreePool(EntryIterator it, Pool pool){
  sanityCheck();
  if(verbose) std::cout << "MemoryManager: Relinquishing " << it->ptr << " of size " << it->bytes << " from " << poolName(pool) << std::endl;
  it->owned_by = nullptr;
  auto &from = getLRUpool(pool);
  auto &to = getFreePool(pool)[it->bytes];
  to.splice(to.end(),from,it);
}
  
void MemoryManager::freeEntry(EntryIterator it, Pool pool){       
  sanityCheck();
  if(pool == DevicePool){
    acceleratorFreeDevice(it->ptr); device_allocated -= it->bytes;
    if(verbose) std::cout << "MemoryManager: Freed device memory " << it->ptr << " of size " << it->bytes << ". Allocated amount is now " << device_allocated << " vs max " << device_pool_max_size << std::endl;
  }else{ //HostPool
    mmap_free(it->ptr, it->bytes);
    host_allocated -= it->bytes;
    if(verbose) std::cout << "MemoryManager: Freed host memory " << it->ptr << " of size " << it->bytes << ". Allocated amount is now " << host_allocated << " vs max " << host_pool_max_size << std::endl;	
  }
}

void MemoryManager::deallocateFreePool(Pool pool, size_t until_allocated_lte){
  sanityCheck();
  size_t &allocated = (pool == DevicePool ? device_allocated : host_allocated);
  if(verbose) std::cout << "MemoryManager: Deallocating free " << poolName(pool) << " until " << until_allocated_lte << " remaining. Current " << allocated << std::endl;
  auto &free_pool = getFreePool(pool);
    
  //Start from the largest    
  auto sit = free_pool.begin();
  while(sit != free_pool.end()){
    auto& entry_list = sit->second;

    auto it = entry_list.begin();
    while(it != entry_list.end()){
      freeEntry(it, pool);
      it = entry_list.erase(it);

      if(allocated <= until_allocated_lte){
	if(entry_list.size() == 0) free_pool.erase(sit); //if we break out after draining the list for a particular size, we need to remove that list from the map
	if(verbose) std::cout << "MemoryManager: deallocateFreePool has freed enough memory" << std::endl;
	return;
      }
    }
    sit = free_pool.erase(sit);
  }
  if(verbose) std::cout << "MemoryManager: deallocateFreePool has freed all of its memory" << std::endl;
}

MemoryManager::EntryIterator MemoryManager::getEntry(size_t bytes, Pool pool){
  sanityCheck();
  if(verbose) std::cout << "MemoryManager: Getting an entry of size " << bytes << " from " << poolName(pool) << std::endl;
  size_t pool_max_size = ( pool == DevicePool ? device_pool_max_size : host_pool_max_size );
  size_t &allocated = (pool == DevicePool ? device_allocated : host_allocated);
    
  if(bytes > pool_max_size) throw std::runtime_error("Requested size is larger than the maximum pool size!");

  auto &LRUpool = getLRUpool(pool);
  auto &free_pool = getFreePool(pool);

  //First check if we have an entry of the right size in the pool
  auto fit = free_pool.find(bytes);
  if(fit != free_pool.end()){
    auto &entry_list = fit->second;
    if(entry_list.size() == 0){
      std::ostringstream os; os << "Free pool for size " << bytes << " bytes has size 0; this pool should have been deleted!";
      throw std::runtime_error(os.str());
    }

    if(entry_list.back().bytes != bytes){
      std::ostringstream os; os << "In pool of size " << bytes << "(check key " << fit->first << ") found entry of size " << entry_list.back().bytes << "! List has size " << entry_list.size() << " and full set of sizes: ";
      for(auto e=entry_list.begin();e!=entry_list.end();e++) os << e->bytes <<  " ";
      os << std::endl;
      throw std::runtime_error(os.str());
    }      
      
    if(verbose) std::cout << "MemoryManager: Found entry " << entry_list.back().ptr << " in free pool" << std::endl;      
    LRUpool.splice(LRUpool.end(), entry_list, std::prev(entry_list.end()));

    if(fit->second.size() == 0) free_pool.erase(fit); //remove the entire, now-empty list
    return std::prev(LRUpool.end());
  }

  //Next, if we have enough room, allocate new memory
  if(allocated + bytes <= pool_max_size){
    if(verbose) std::cout << "MemoryManager: Allocating new memory for entry" << std::endl;
    return allocEntry(bytes, pool);
  }

  //Next, we should free up unused blocks from the free pool
  if(verbose) std::cout << "MemoryManager: Clearing up space from the free pool to make room" << std::endl;
  deallocateFreePool(pool, pool_max_size - bytes);
  if(allocated + bytes <= pool_max_size){
    if(verbose) std::cout << "MemoryManager: Allocating new memory for entry" << std::endl;
    return allocEntry(bytes, pool);
  }

  //Evict old data until we have enough room
  //If we hit an entry with just the right size, reuse the pointer
  if(verbose) std::cout << "MemoryManager: Evicting data to make room" << std::endl;
  auto it = LRUpool.begin();
  while(it != LRUpool.end()){
    if(verbose) std::cout << "MemoryManager: Attempting to evict entry " << it->ptr << std::endl;

    if(it->owned_by->lock_entry){ //don't evict an entry that is currently in use
      if(verbose) std::cout << "MemoryManager: Entry is assigned to an open view or prefetch for handle " << it->owned_by << ", skipping" << std::endl;
      ++it;
      continue;
    }

    bool erase = true;
    void* reuse;
    if(it->bytes == bytes){
      if(verbose) std::cout << "MemoryManager: Found entry " << it->ptr << " has the right size, yoink" << std::endl;
      reuse = it->ptr;
      erase = false;
    }
    it = evictEntry(it, erase, pool);

    if(!erase){
      if(verbose) std::cout << "MemoryManager: Reusing memory " << reuse << std::endl;
      //reuse existing allocation
      Entry e;
      e.bytes = bytes;
      e.ptr = reuse;
      e.owned_by = nullptr;
      return LRUpool.insert(LRUpool.end(),e);
    }else if(allocated + bytes <= pool_max_size){ //allocate if we have enough room
      if(verbose) std::cout << "MemoryManager: Memory available " << allocated << " is now sufficient, allocating" << std::endl;
      return allocEntry(bytes,pool);
    }
  }

  std::ostringstream ss; ss << "MemoryManager was not able to get an entry for " << bytes << " bytes";  
  throw std::runtime_error(ss.str());
}

void MemoryManager::attachEntry(Handle &handle, Pool pool){
  sanityCheck();
  if(pool == DevicePool){
    if(handle.device_valid) throw std::runtime_error("Expect !handle.device_valid");
    handle.device_entry = getEntry(handle.bytes, DevicePool);
    handle.device_valid = true;
    handle.device_entry->owned_by = &handle;
  }else{
    if(handle.host_valid) throw std::runtime_error("Expect !handle.host_valid");
    handle.host_entry = getEntry(handle.bytes, HostPool);
    handle.host_valid = true;
    handle.host_entry->owned_by = &handle;
  }
}

void MemoryManager::touchEntry(Handle &handle, Pool pool){
  sanityCheck();
  EntryIterator entry = pool == DevicePool ? handle.device_entry : handle.host_entry;
  if(verbose) std::cout << "MemoryManager: Touching entry " << entry->ptr << " in " << poolName(pool) << std::endl;
  auto &p = getLRUpool(pool);
  p.splice(p.end(),p,entry); //doesn't invalidate any iterators :)
}

void MemoryManager::syncDeviceToHost(Handle &handle){
  sanityCheck();
  if(!handle.initialized) throw std::runtime_error("Attempting to sync an uninitialized data region!");
  if(!handle.host_in_sync){
    if(!handle.device_in_sync || !handle.device_valid) throw std::runtime_error("Device copy is either not in sync or invalid");
    if(!handle.host_valid) attachEntry(handle, HostPool);
    if(verbose) std::cout << "MemoryManager: Synchronizing device " << handle.device_entry->ptr << " to host " << handle.host_entry->ptr << std::endl;
    acceleratorCopyFromDevice(handle.host_entry->ptr, handle.device_entry->ptr, handle.bytes);
    handle.host_in_sync = true;
  }
}
void MemoryManager::syncHostToDevice(Handle &handle){
  sanityCheck();
  if(!handle.initialized) throw std::runtime_error("Attempting to sync an uninitialized data region!");
  if(!handle.device_in_sync){
    if(!handle.host_in_sync || !handle.host_valid) throw std::runtime_error("Host copy is either not in sync or invalid");
    if(!handle.device_valid) attachEntry(handle, DevicePool);
    if(verbose) std::cout << "MemoryManager: Synchronizing host " << handle.host_entry->ptr << " to device " << handle.device_entry->ptr << std::endl;
    acceleratorCopyToDevice(handle.device_entry->ptr, handle.host_entry->ptr, handle.bytes);
    handle.device_in_sync = true;
  }
}  

void MemoryManager::syncHostToDisk(Handle &handle){
  sanityCheck();
  if(!handle.initialized) throw std::runtime_error("Attempting to sync an uninitialized data region!");
  if(!handle.disk_in_sync){
    if(!handle.host_in_sync || !handle.host_valid) throw std::runtime_error("Host copy is either not in sync or invalid");

    auto time = now();

    static size_t idx = 0;
    if(handle.disk_file == ""){
      handle.disk_file = disk_root + "/mempool." + std::to_string(UniqueID()) + "." + std::to_string(idx++);
    }
    if(verbose) std::cout << "MemoryManager: Synchronizing host " << handle.host_entry->ptr << " to disk " << handle.disk_file << std::endl;

    if(!handle.disk_file_exists){
      local_disk_allocated += handle.bytes;
      local_disk_allocated_HWM = std::max(local_disk_allocated_HWM, local_disk_allocated);
    }

    std::fstream f(handle.disk_file.c_str(), std::ios::out | std::ios::binary);
    if(!f.good()){
      std::ostringstream ss; ss << "Failed to open file " << handle.disk_file << " for write";
      throw std::runtime_error(ss.str());
    }
    f.write((char*)handle.host_entry->ptr, handle.bytes);
    if(!f.good()){
      std::ostringstream ss; ss << "Write error in file " << handle.disk_file << " for write";
      throw std::runtime_error(ss.str());
    }
    f.flush(); //should ensure data is written to disk immediately and not kept around in some memory buffer, but may slow things down
          
    handle.disk_file_exists = true;
    handle.disk_in_sync = true;

    double dt = since(time);
    if(verbose){
      double MB = byte_to_MB(handle.bytes);
      std::cout << "MemoryManager: Wrote " << MB << "MB in " << dt << "s, rate " << MB/dt << " MB/s" << std::endl;
    }
    if(this->io_logger){
      double MB = byte_to_MB(handle.bytes);
      (*io_logger) << usSinceEpoch() << " : Evict " << MB << " MB in " << dt << "s, rate " << MB/dt << " MB/s, to file " << handle.disk_file << std::endl << std::flush;
    }
  }
}

void MemoryManager::removeDiskData(Handle &handle, bool in_memory_check){
  if(!handle.disk_file_exists) return;
  if(in_memory_check && !handle.host_in_sync && !handle.device_in_sync) throw std::runtime_error("Cannot delete disk data while there is no in-memory copy");

  if(verbose) std::cout << "MemoryManager: Erasing cache file " << handle.disk_file << std::endl;
  
  if(remove(handle.disk_file.c_str()) != 0){
    std::ostringstream os; os << "Disk data removal for file " << handle.disk_file << " failed for reason " << strerror(errno);
    throw std::runtime_error(os.str());
  }
  this->local_disk_allocated -= handle.bytes;
  
  handle.disk_file_exists = false;
  handle.disk_in_sync = false;
}

void MemoryManager::syncDiskToHost(Handle &handle){
  sanityCheck();
  if(!handle.initialized) throw std::runtime_error("Attempting to sync an uninitialized data region!");
  if(!handle.host_in_sync){
    if(!handle.disk_in_sync || !handle.disk_file_exists){
      std::ostringstream os; os << "Rank " << UniqueID() << ", disk copy " << handle.disk_file << " is either not in sync or invalid: disk_in_sync=" << handle.disk_in_sync << " disk_file_exists=" << handle.disk_file_exists;      
      throw std::runtime_error(os.str());
    }
    if(!handle.host_valid) attachEntry(handle, HostPool);     

    auto time = now();
    
    if(verbose) std::cout << "MemoryManager: Synchronizing disk " << handle.disk_file << " to host " << handle.host_entry->ptr << std::endl;

    std::fstream f(handle.disk_file.c_str(), std::ios::in | std::ios::binary);
    if(!f.good()){
      std::ostringstream os; os << "Failed to open file " << handle.disk_file << " for read";
      throw std::runtime_error(os.str());
    }
    f.read((char*)handle.host_entry->ptr, handle.bytes);
    if(!f.good()){
      std::ostringstream os; os << "Read error in file " << handle.disk_file;
      throw std::runtime_error(os.str());
    }

    handle.host_in_sync = true;

    double dt = since(time);
    if(verbose){    
      double MB = byte_to_MB(handle.bytes);
      std::cout << "MemoryManager: Read " << MB << "MB in " << dt << "s, rate " << MB/dt << " MB/s" << std::endl;
    }
    if(this->io_logger){
      double MB = byte_to_MB(handle.bytes);
      (*io_logger) << usSinceEpoch() << " : Restore " << MB << " MB in " << dt << "s, rate " << MB/dt << " MB/s, from file " << handle.disk_file << std::endl << std::flush;
    }

    //optionally delete no-longer-needed cache file (if we don't we might be able to avoid additional writes if the disk copy stays in-sync providing we have enough disk space to allow for it)
    if(this->delete_local_diskdata_on_restore) removeDiskData(handle); 
  }
}

void MemoryManager::syncForRead(Handle &handle, Pool pool){
  sanityCheck();
  if(pool == HostPool){    
    if(!handle.host_in_sync){
      if(handle.device_in_sync) syncDeviceToHost(handle);
      else if(handle.disk_in_sync) syncDiskToHost(handle);
      else if(handle.initialized) throw std::runtime_error("Data has been initialized but no active copy!");
      //Allow copies from uninitialized data, eg in copy constructor called during initialization of vector of fields
    }
  }else{ //DevicePool
    if(handle.device_prefetch_underway) throw std::runtime_error("Attempting to open a device read view while a prefetch is still underway!");

    if(!handle.device_in_sync){
      if(handle.host_in_sync) syncHostToDevice(handle);
      else if(handle.disk_in_sync){
	syncDiskToHost(handle);
	syncHostToDevice(handle);
      }
      else if(handle.initialized) throw std::runtime_error("Data has been initialized but no active copy!");
    }
  }
}

void MemoryManager::markForWrite(Handle &handle, Pool pool){
  sanityCheck();
  if(pool == HostPool){
    handle.host_in_sync = true;
    handle.device_in_sync = false;
    handle.disk_in_sync = false;
    handle.initialized = true;
  }else{ //DevicePool
    if(handle.device_prefetch_underway) throw std::runtime_error("Attempting to open a device write view while a prefetch is still underway!");

    handle.host_in_sync = false;
    handle.device_in_sync = true;
    handle.disk_in_sync = false;
    handle.initialized = true;
  }
}  

void MemoryManager::prepareEntryForView(Handle &handle, Pool pool){
  sanityCheck();
  bool valid = pool == DevicePool ? handle.device_valid : handle.host_valid;
  if(!valid) attachEntry(handle,pool);
  else touchEntry(handle, pool); //move to end of LRU
}
 
MemoryManager::EntryIterator MemoryManager::evictEntry(EntryIterator entry, bool free_it, Pool pool){
  sanityCheck();
  if(verbose) std::cout << "MemoryManager: Evicting entry " << entry->ptr << " from " << poolName(pool) << std::endl;
	
  if(entry->owned_by != nullptr){
    if(verbose) std::cout << "MemoryManager: Entry is owned by handle " << entry->owned_by << ", detaching" << std::endl;
    Handle &handle = *entry->owned_by;
    if(handle.lock_entry) throw std::runtime_error("Cannot evict a locked entry!");

    if(pool == DevicePool){
      //Copy data back to host if not in sync
      if(handle.device_in_sync && !handle.host_in_sync){
	if(verbose) std::cout << "MemoryManager: Host is not in sync with device, copying back before detach" << std::endl;
	syncDeviceToHost(handle);
      }
      handle.device_entry->owned_by = nullptr;
      handle.device_in_sync = false;      
      handle.device_valid = false; //evict
    }else{
      //Copy data to disk if not in sync
      if(handle.host_in_sync && !handle.disk_in_sync){
	if(verbose) std::cout << "MemoryManager: Disk is not in sync with device, copying back before detach" << std::endl;
	syncHostToDisk(handle);
      }
      handle.host_entry->owned_by = nullptr;
      handle.host_in_sync = false;
      handle.host_valid = false; //evict
    }    
  }
  if(free_it) freeEntry(entry, pool); //deallocate the memory entirely (optional, we might want to reuse it)
  return getLRUpool(pool).erase(entry); //remove from list
}

void MemoryManager::summarizePoolStatus(std::ostream &os, const std::string &descr, const std::map<size_t,std::list<Entry>, std::greater<size_t> > &pool_stat){
  os << descr << " (size_MB,count,total_MB):";
  double tot = 0;
  for(auto const &e : pool_stat){
    double MB = byte_to_MB(e.first);
    os << " (" << MB << "," << e.second.size() << "," << e.second.size() * MB << ")";
    tot += e.second.size() * MB;
  }
  os << " : TOTAL " << tot << std::endl;
}
void MemoryManager::summarizePoolStatus(std::ostream &os, const std::string &descr, const std::map<size_t,int, std::greater<size_t> > &pool_stat){
  os << descr << " (size_MB,count,total_MB):";
  double tot = 0;
  for(auto const &e : pool_stat){
    double MB = byte_to_MB(e.first);
    os << " (" << MB << "," << e.second << "," << e.second * MB << ")";
    tot += e.second * MB;
  }
  os << " : TOTAL " << tot << std::endl;
}

MemoryManager::~MemoryManager(){
  std::cout << "~MemoryManager handles.size()=" << handles.size() << " device_in_use_pool.size()=" << device_in_use_pool.size() << " host_in_use_pool.size()=" << host_in_use_pool.size() << std::endl;
  auto it = device_in_use_pool.begin();
  while(it != device_in_use_pool.end()){
    freeEntry(it, DevicePool);
    it = device_in_use_pool.erase(it);
  }
  it = host_in_use_pool.begin();
  while(it != host_in_use_pool.end()){
    freeEntry(it, HostPool);
    it = host_in_use_pool.erase(it);
  }
  deallocateFreePool(HostPool);
  deallocateFreePool(DevicePool);
  if(this->io_logger) delete this->io_logger;
}

size_t MemoryManager::getDiskCachedBytes() const{
  size_t out = 0;
  for(auto const &h : handles)
    if(h.initialized && !h.device_in_sync && !h.host_in_sync && h.disk_file.size() )
      out += h.bytes;
  return out;
}

size_t MemoryManager::getDiskUsedBytes() const{
  return local_disk_allocated;
}

std::string MemoryManager::report(bool detailed) const{
  std::ostringstream os;
  os << "MemoryManager consumption - device: " << byte_to_MB(device_allocated)
     << " MB, host: " << byte_to_MB(host_allocated)
     << " MB, disk (cached): " << byte_to_MB(getDiskCachedBytes())
     << " MB, disk (total): " << byte_to_MB(local_disk_allocated);
  os << "MemoryManager HWM - device: " << byte_to_MB(device_allocated_HWM)
     << " MB, host: " << byte_to_MB(host_allocated_HWM)
     << " MB, disk (total): " << byte_to_MB(local_disk_allocated_HWM);
  
  if(detailed){
    os << std::endl;

    summarizePoolStatus(os, "DeviceFreePool", device_free_pool);
    std::map<size_t, int, std::greater<size_t> > in_use;
    for(auto const &e : device_in_use_pool){
      auto it = in_use.find(e.bytes);
      if(it == in_use.end()){
	in_use[e.bytes] = 1;
      }else{
	++(it->second);
      }
    }
    summarizePoolStatus(os, "DeviceInUsePool", in_use);
    in_use.clear();
    summarizePoolStatus(os, "HostFreePool", host_free_pool);
    for(auto const &e : host_in_use_pool){
      auto it = in_use.find(e.bytes);
      if(it == in_use.end()){
	in_use[e.bytes] = 1;
      }else{
	++(it->second);
      }
    }
    summarizePoolStatus(os, "HostInUsePool", in_use);
  }

  return os.str();
}

void MemoryManager::evictToDisk(HandleIterator h){
  if(verbose) std::cout << "MemoryManager: manually evicting data of size " << h->bytes << std::endl;
  if(h->device_valid) evictEntry(h->device_entry, true, DevicePool);
  if(h->host_valid) evictEntry(h->host_entry, true, HostPool);
  if(!h->disk_in_sync) throw std::runtime_error("Logic bomb: after eviction, disk data not in sync!");
}

//Allocate data, initially in the specified pool
MemoryManager::HandleIterator MemoryManager::allocate(size_t bytes, Pool pool){
  if(in_thread_parallel_region()) throw std::runtime_error("Cannot call in thread parallel region");
  
  sanityCheck();
  if(verbose) std::cout << "MemoryManager: Request for allocation of size " << bytes << " in " << poolName(pool) << std::endl;

  HandleIterator it = handles.insert(handles.end(),Handle());
  Handle &h = *it;
  h.bytes = bytes;
  attachEntry(h,pool);    
  return it;
}

//Open a view of a specific type, returning a raw pointer to the data
void* MemoryManager::openView(ViewMode mode, HandleIterator h){
  if(in_thread_parallel_region()) throw std::runtime_error("Cannot call in thread parallel region");
  
  sanityCheck();
  ++h->lock_entry; //make sure it isn't evicted!
  Pool pool = (mode == HostRead || mode == HostWrite || mode == HostReadWrite) ? HostPool : DevicePool;   
  prepareEntryForView(*h,pool); 
  bool read(false), write(false);
    
  switch(mode){
  case HostRead:
  case DeviceRead:
    read=true; break;
  case HostWrite:
  case DeviceWrite:
    write=true; break;
  case HostReadWrite:
  case DeviceReadWrite:
    write=read=true; break;
  }

  if(read) syncForRead(*h,pool);
  if(write) markForWrite(*h,pool);

  return pool == HostPool ? h->host_entry->ptr : h->device_entry->ptr;
}  

void MemoryManager::closeView(HandleIterator h){
  if(in_thread_parallel_region()) throw std::runtime_error("Cannot call in thread parallel region");
  if(h->lock_entry == 0) throw std::runtime_error("Lock state has already been decremented to 0; this should not happen");
  --h->lock_entry;
}

// void MemoryManager::enqueuePrefetch(ViewMode mode, HandleIterator h){    
//   if(omp_in_parallel()) ERR.General("MemoryManager","enqueuePrefetch","Cannot call in OMP parallel region");
//   sanityCheck();
//   if(mode == HostRead || mode == HostReadWrite){
//     //no support for device->host async copies yet
//   }else if( (mode == DeviceRead || mode == DeviceReadWrite) && h->host_valid && h->host_in_sync){ //only start a prefetch if the host memory region is attached and up-to-date
//     prepareEntryForView(*h,DevicePool); 
//     if(!h->device_in_sync){
//       asyncTransferManager::globalInstance().enqueue(h->device_entry->ptr,h->host_entry->ptr,h->bytes);
//       h->device_in_sync = true; //technically true only if the prefetch is complete; make sure to wait!!
//       ++h->lock_entry; //use this flag also for prefetches to ensure the memory region is not evicted while the async copy is happening
//       h->device_prefetch_underway = true; //mark for error checking
//       device_queued_prefetches.push_back(h);
//     }
//   }
// }

// void MemoryManager::startPrefetches(){
//   if(omp_in_parallel()) ERR.General("MemoryManager","startPrefetches","Cannot call in OMP parallel region");
//   sanityCheck();
//   if(device_queued_prefetches.size()==0) return;
//   asyncTransferManager::globalInstance().start();
// }

// void MemoryManager::waitPrefetches(){
//   if(omp_in_parallel()) ERR.General("MemoryManager","waitPrefetches","Cannot call in OMP parallel region");
//   sanityCheck();
//   if(device_queued_prefetches.size()==0) return;   
//   asyncTransferManager::globalInstance().wait();
//   for(auto h : device_queued_prefetches){
//     if(h->lock_entry == 0) ERR.General("MemoryManager","waitPrefetches","lock_entry has already been decremented to 0; this should not happen");
//     if(!h->device_prefetch_underway) ERR.General("MemoryManager","waitPrefetches","device_prefetch_underway has already been decremented to 0; this should not happen");
//     --h->lock_entry; //unlock
//     h->device_prefetch_underway = false;
//   }
//   device_queued_prefetches.clear();
// }

void MemoryManager::free(HandleIterator h){
  if(in_thread_parallel_region()) throw std::runtime_error("Cannot call in thread parallel region");
  sanityCheck();
  if(h->lock_entry) throw std::runtime_error("Attempting to free locked entry");

  if(h->device_valid) moveEntryToFreePool(h->device_entry, DevicePool);
  if(h->host_valid) moveEntryToFreePool(h->host_entry, HostPool);
  removeDiskData(*h, false);
  handles.erase(h);
}
