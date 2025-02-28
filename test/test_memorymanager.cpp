#include<HPCortex.hpp>
#include<fcntl.h>
#include<unistd.h>

void testMemoryManager(){
  size_t MB = 1024*1024;
  size_t max_size = 3*MB;

  {
    std::cout << "TEST: Test device pool eviction to host" << std::endl;
    MemoryManager pool(max_size, max_size);  
    pool.setVerbose(true);

    //Allocate full size on device
    auto h1 = pool.allocate(max_size, MemoryManager::DevicePool);
    assert(h1->device_valid);
    assert(pool.getAllocatedBytes(MemoryManager::DevicePool) == max_size);

    //Touch the device data to mark device side in sync
    {
      void* p = pool.openView(DeviceWrite,h1);
      acceleratorMemSet(p,0x1A,max_size);
      pool.closeView(h1);
    }

    //Try to allocate again on device, it should evict the current entry to host
    auto h2 = pool.allocate(max_size, MemoryManager::DevicePool);
    assert(h2->device_valid);
    assert(pool.getAllocatedBytes(MemoryManager::DevicePool) == max_size);

    assert(!h1->device_valid);
    assert(h1->host_valid);
    assert(pool.getAllocatedBytes(MemoryManager::HostPool) == max_size);
    {
      char* p = (char*)pool.openView(HostRead,h1);
      assert(*p == 0x1A);
      pool.closeView(h1);
    }
    pool.free(h1);
    pool.free(h2);
  }

  {
    std::cout << "TEST: Test host pool eviction to disk" << std::endl;
    MemoryManager pool(max_size, max_size);  
    pool.setVerbose(true);

    //Allocate full size on host
    auto h1 = pool.allocate(max_size, MemoryManager::HostPool);
    assert(h1->host_valid);
    assert(pool.getAllocatedBytes(MemoryManager::HostPool) == max_size);

    //Touch the host data to mark host side in sync
    {
      void* p = pool.openView(HostWrite,h1);
      memset(p,0x1A,max_size);
      pool.closeView(h1);
    }

    //Try to allocate again on host, it should evict the current entry to disk
    auto h2 = pool.allocate(max_size, MemoryManager::HostPool);
    assert(h2->host_valid);
    assert(pool.getAllocatedBytes(MemoryManager::HostPool) == max_size);

    assert(!h1->host_valid);
    assert(h1->disk_in_sync);
    {
      char* p = (char*)pool.openView(HostRead,h1); //pull back from disk, should also evict h2
      assert(*p == 0x1A);
      pool.closeView(h1);
    }
    assert(!h2->host_valid);
    //assert(h2->disk_in_sync); //actually no, because data was never written to h2 there is no need to store it when evicted

    std::cout << "TEST: Check (manually) that the files are deleted:" << std::endl;
    pool.free(h1);
    //pool.free(h2);     
  }

  {
    std::cout << "TEST: Test manual eviction to disk" << std::endl;
    MemoryManager pool(max_size, max_size);  
    pool.setVerbose(true);

    int data[100];
    for(int i=0;i<100;i++) data[i] = 3*i-1;
    size_t dsize = 100*sizeof(int);

    auto h1 = pool.allocate(dsize, MemoryManager::DevicePool);

    //Put the data on the top level (device)
    {
      void* p = pool.openView(DeviceWrite,h1);
      acceleratorCopyToDevice(p, data, dsize);
      pool.closeView(h1);
    }

    //Force evict
    pool.evictToDisk(h1);
    assert(!h1->device_valid);
    assert(!h1->host_valid);
    assert(h1->disk_in_sync);
    
    //Read back on host
    {
      int* p = (int*)pool.openView(HostRead,h1);
      for(int i=0;i<100;i++) assert(p[i] == data[i]);
      pool.closeView(h1);
    }
    pool.free(h1);
  }    
 
  {
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "TEST: Test deletion of disk data on restore" << std::endl;
    MemoryManager pool(max_size, max_size);  
    pool.setVerbose(true);
    pool.enableDeletionOfLocalDiskDataOnRestore(true);

    int data[100];
    for(int i=0;i<100;i++) data[i] = 3*i-UniqueID();
    size_t dsize = 100*sizeof(int);

    //Test first for local data
    auto h1 = pool.allocate(dsize, MemoryManager::HostPool);
    {
      void* p = pool.openView(HostWrite,h1);
      memcpy(p, data, dsize);
      pool.closeView(h1);
    }
    pool.evictToDisk(h1);
    
    assert(h1->disk_file_exists && h1->disk_in_sync);
    {
      int r = open(h1->disk_file.c_str(), O_RDONLY);
      assert(r != -1);
      close(r);
    }
    //Restore
    {
      int* p = (int*)pool.openView(HostRead,h1);
      bool fail = false;
      for(int i=0;i<100;i++){
	if(p[i] != data[i]){
	  printf("Rank %d idx %d got %d expected %d\n", UniqueID(), i, p[i], data[i]);
	  fail = true;
	}
      }
      assert(!fail);
      pool.closeView(h1);
    }
    //Ensure file doesn't exist
    assert(!h1->disk_file_exists);
    {
      int r = open(h1->disk_file.c_str(), O_RDONLY);
      assert(r == -1);
      close(r);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    pool.free(h1);
  }    

  std::cout << "testMemoryManager passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testMemoryManager();

  return 0;
}
