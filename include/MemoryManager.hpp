#pragma once
#include<list>
#include<string>
#include<map>
#include<fstream>

enum ViewMode { HostRead, HostWrite, DeviceRead, DeviceWrite, HostReadWrite, DeviceReadWrite };

//3-level memory pool manager with device,host,disk storage locations with eviction possible from device, host
class MemoryManager{
public:
  struct Handle;

  struct Entry{ //A memory region of a specific size
    size_t bytes;
    void* ptr;
    Handle* owned_by; //if attached, set to attached handle
  };
  typedef std::list<Entry>::iterator EntryIterator;

  struct Handle{
    size_t lock_entry; //if >0 , eviction is disabled 

    bool device_valid; //device entry is attached; set false upon eviction
    EntryIterator device_entry; //memory region on device

    bool host_valid; //host entry is attached; set false upon eviction
    EntryIterator host_entry; //memory region on host

    size_t bytes;

    bool device_in_sync; //device copy (if exists) is up-to-date
    bool host_in_sync; //host copy (if exists) is up-to-date
    bool disk_in_sync; //disk copy (if exists) is up-to-date
    std::string disk_file;
    bool disk_file_exists; //does the disk file exist?

    bool device_prefetch_underway; //allow for error checking if try to open device read view with a pending prefetch

    bool initialized; //keep track of whether the data has had a write

    Handle(): host_in_sync(false), device_in_sync(false), disk_in_sync(false), lock_entry(0), device_valid(false), host_valid(false), 
      bytes(0), disk_file(""), initialized(false), device_prefetch_underway(false), disk_file_exists(false){ }
  };

  typedef std::list<Handle>::iterator HandleIterator;

  enum Pool { DevicePool, HostPool };
  
protected:
  bool verbose;
  std::ofstream* io_logger;
  std::list<Handle> handles; //managed objects currently in use

  std::list<Entry> device_in_use_pool; //LRU
  std::map<size_t,std::list<Entry>, std::greater<size_t> > device_free_pool; //sorted by size in descending order
  std::list<HandleIterator> device_queued_prefetches; //track open prefetches to device from host

  std::list<Entry> host_in_use_pool; //LRU
  std::map<size_t,std::list<Entry>, std::greater<size_t> > host_free_pool; //sorted by size in descending order
  std::list<HandleIterator> host_queued_prefetches; //track open prefetches to host from disk

  inline std::list<Entry> & getLRUpool(Pool pool){ return pool == DevicePool ? device_in_use_pool : host_in_use_pool; }
  inline std::map<size_t,std::list<Entry>, std::greater<size_t> > & getFreePool(Pool pool){ return pool == DevicePool ? device_free_pool : host_free_pool; }
  inline std::string poolName(Pool pool){ return pool == DevicePool ? "DevicePool" : "HostPool"; }   
  
  size_t device_allocated;
  size_t host_allocated;
  size_t device_pool_max_size;
  size_t host_pool_max_size;
  size_t local_disk_allocated; //amount of space currently taken by caches for this rank's data

  //High water marks for the above
  size_t device_allocated_HWM;
  size_t host_allocated_HWM;
  size_t local_disk_allocated_HWM;

  std::string disk_root; //root location for temp files, default "."
  
  //optionally delete no-longer-needed disk cache files when restoring
  //disabled by default because we might be able to avoid additional writes if the disk copy stays in-sync providing we have enough disk space to allow for it
  bool delete_local_diskdata_on_restore;

  //Allocate a new entry of the given size and move to the end of the LRU queue, returning a pointer
  EntryIterator allocEntry(size_t bytes, Pool pool);

  void sanityCheck();

  //Relinquish the entry from the LRU and put in the free pool
  void moveEntryToFreePool(EntryIterator it, Pool pool);
  
  //Free the memory associated with an entry
  void freeEntry(EntryIterator it, Pool pool);
  
  void deallocateFreePool(Pool pool, size_t until_allocated_lte = 0);

  //Get an entry either new or from the pool
  //It will automatically be moved to the end of the in_use_pool list
  EntryIterator getEntry(size_t bytes, Pool pool);

  void attachEntry(Handle &handle, Pool pool);

  //Move the entry to the end and return a new iterator
  void touchEntry(Handle &handle, Pool pool);
  
  void syncDeviceToHost(Handle &handle);

  void syncHostToDevice(Handle &handle);

  void syncHostToDisk(Handle &handle);

  void syncDiskToHost(Handle &handle);

  //Perform operations required to ensure data is available to read in the given pool
  void syncForRead(Handle &handle, Pool pool);

  //Perform operations required to ensure data is available to write in the given pool
  void markForWrite(Handle &handle, Pool pool);
 
  //Ensure an entry is prepared in the given pool for read/write operations and move the entry to the end of the LRU queue
  void prepareEntryForView(Handle &handle, Pool pool);
 
  //Evict an entry (with optional freeing of associated memory), and return an entry to the next item in the LRU
  EntryIterator evictEntry(EntryIterator entry, bool free_it, Pool pool);

  //Remove the data on disk. If in_memory_check==true, check to ensure an in-memory copy exists, throwing an error if not
  void removeDiskData(Handle &handle, bool in_memory_check = true);

  static void summarizePoolStatus(std::ostream &os, const std::string &descr, const std::map<size_t,std::list<Entry>, std::greater<size_t> > &pool_stat);
  static void summarizePoolStatus(std::ostream &os, const std::string &descr, const std::map<size_t,int, std::greater<size_t> > &pool_stat);

public:

  MemoryManager(): device_allocated(0), device_pool_max_size(1024*1024*1024), host_allocated(0),
		   host_pool_max_size(1024*1024*1024), local_disk_allocated(0),  verbose(false), disk_root("."),
		   delete_local_diskdata_on_restore(false), 
		   device_allocated_HWM(0), host_allocated_HWM(0), local_disk_allocated_HWM(0), io_logger(nullptr)  {}

  MemoryManager(size_t max_size_device, size_t max_size_host): MemoryManager(){
    device_pool_max_size = max_size_device;
    host_pool_max_size = max_size_host;
  }

  ~MemoryManager();

  void setVerbose(bool to){ verbose = to; }

  //If enabled, each rank outputs logging information about IO activities (for eviction/restoration) to a separate file in the working directory
  void enableIOlogging();

  //Set the root path for scratch data
  void setDiskRoot(const std::string &to){ disk_root = to; }
  
  //Get the root path for scratch data
  const std::string &getDiskRoot() const{ return disk_root; }

  //Optionally delete no-longer-needed disk cache files when restoring
  //Disabled by default because we might be able to avoid additional writes if the disk copy stays in-sync providing we have enough disk space to allow for it
  void enableDeletionOfLocalDiskDataOnRestore(bool val = true){ delete_local_diskdata_on_restore = val; }

  //Set the pool max size in bytes. When the next eviction cycle happens the extra memory will be deallocated
  void setPoolMaxSize(size_t to, Pool pool){
    auto &m = (pool == DevicePool ? device_pool_max_size : host_pool_max_size );
    m = to;
  }
  //Get the amount of data allocated in a given pool
  size_t getAllocatedBytes(Pool pool) const{ return pool == DevicePool ? device_allocated : host_allocated; }

  //Return the amount of data (in bytes) associated with objects whose only representation is on disk
  size_t getDiskCachedBytes() const;

  //Return the total disk usage for process-local data (i.e. even if the disk representation is out of date or not the primary copy)
  size_t getDiskUsedBytes() const;

  std::string report(bool detailed = false) const;

  //Forcefully evict the data associated with the handle from all layers down to the disk, freeing the memory
  //(data can of course be recovered by opening a view)
  void evictToDisk(HandleIterator h);

  //Allocate data, initially in the specified pool
  HandleIterator allocate(size_t bytes, Pool pool = DevicePool);

  //Open a view of a specific type, returning a raw pointer to the data
  void* openView(ViewMode mode, HandleIterator h);

  void closeView(HandleIterator h);

  // void enqueuePrefetch(ViewMode mode, HandleIterator h);

  // void startPrefetches();

  // void waitPrefetches();

  void free(HandleIterator h);

  //Return the number of managed objects still active (unfreed)
  inline size_t nOpenHandles() const{ return handles.size(); }

  inline static MemoryManager & globalPool(){
    static MemoryManager pool;
    return pool;
  }      

  //Prevent eviction of an object, use with care
  inline void lock(HandleIterator h){ ++h->lock_entry; }
  inline void unlock(HandleIterator h){ --h->lock_entry; }  
};

inline std::string memPoolManagerReport(bool detailed = false){
  return MemoryManager::globalPool().report(detailed);
}
