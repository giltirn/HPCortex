#include <HPCortex.hpp>
#include <ManagedArray.hpp>

void testManagedArray(){
  ///////////////////////////////////// INITIALIZATION/ACCESS /////////////////////////
  
  {
    //Test host alloc with initial val
    size_t sz = 20;
    ManagedArray<double> v(sz, 0.31, MemoryManager::Pool::HostPool);

    autoView(vv, v, HostRead);
    for(int i=0;i<sz;i++)
      assert(vv[i] == 0.31);
  }
  {
    //Test host alloc with assigned
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::HostPool);

    {
      autoView(vv, v, HostWrite);
      for(int i=0;i<sz;i++)
	vv[i] = 0.31;
    }
    {
      autoView(vv, v, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }
  {
    //Test device alloc with initial val
    size_t sz = 20;
    ManagedArray<double> v(sz, 0.31, MemoryManager::Pool::DevicePool);

    autoView(vv, v, HostRead);
    for(int i=0;i<sz;i++)
      assert(vv[i] == 0.31);
  }
  {
    //Test device write, host read
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::DevicePool);

    {
      autoView(vv, v, DeviceWrite);
      accelerator_for(i, sz, {	  
	  vv[i] = 0.31;
	});
    }
    {
      autoView(vv, v, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }

  ///////////////////////////////////// CONSTRUCTORS ///////////////////////////////////////
  
  {
    //Test move constructor with data on host
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::HostPool);

    {
      autoView(vv, v, HostWrite);
      for(int i=0;i<sz;i++)
	vv[i] = 0.31;
    }

    ManagedArray<double> v2(std::move(v));
    assert(v.size() == 0);
    assert(v2.size() == sz);
        
    {
      autoView(vv, v2, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }
  {
    //Test move constructor with data on device
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::DevicePool);

    {
      autoView(vv, v, DeviceWrite);
      accelerator_for(i, sz, {	  
	  vv[i] = 0.31;
	});
    }

    ManagedArray<double> v2(std::move(v));
    assert(v.size() == 0);
    assert(v2.size() == sz);
        
    {
      autoView(vv, v2, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }

  {
    //Test copy constructor with data on host
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::HostPool);

    {
      autoView(vv, v, HostWrite);
      for(int i=0;i<sz;i++)
	vv[i] = 0.31;
    }

    ManagedArray<double> v2(v);
    assert(v.size() == sz);
    assert(v2.size() == sz);
        
    {
      autoView(vv, v2, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }
  {
    //Test copy constructor with data on device
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::DevicePool);

    {
      autoView(vv, v, DeviceWrite);
      accelerator_for(i, sz, {	  
	  vv[i] = 0.31;
	});
    }

    ManagedArray<double> v2(v);
    assert(v.size() == sz);
    assert(v2.size() == sz);
        
    {
      autoView(vv, v2, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }
  {
    //Test vector constructor
    size_t sz = 20;
    std::vector<double> init(sz);
    for(int i=0;i<sz;i++) init[i] = 3.141*i;
    
    ManagedArray<double> v(init);
        
    {
      autoView(vv, v, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 3.141*i);
    }
  }

  /////////////////////////////////// ASSIGNMENT ///////////////////////////////////

  {
    //Test move assignment with data on host
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::HostPool);

    {
      autoView(vv, v, HostWrite);
      for(int i=0;i<sz;i++)
	vv[i] = 0.31;
    }

    ManagedArray<double> v2(11);
    v2 = std::move(v);
    assert(v.size() == 0);
    assert(v2.size() == sz);
        
    {
      autoView(vv, v2, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }
  {
    //Test move assignment with data on device
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::DevicePool);

    {
      autoView(vv, v, DeviceWrite);
      accelerator_for(i, sz, {	  
	  vv[i] = 0.31;
	});
    }

    ManagedArray<double> v2(11);
    v2 = std::move(v);
    assert(v.size() == 0);
    assert(v2.size() == sz);
        
    {
      autoView(vv, v2, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }

  {
    //Test copy assignment with data on host
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::HostPool);

    {
      autoView(vv, v, HostWrite);
      for(int i=0;i<sz;i++)
	vv[i] = 0.31;
    }

    ManagedArray<double> v2(11,0.1);
    v2 = v;
    assert(v.size() == sz);
    assert(v2.size() == sz);
        
    {
      autoView(vv, v2, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }
  {
    //Test copy assignment with data on device
    size_t sz = 20;
    ManagedArray<double> v(sz, MemoryManager::Pool::DevicePool);

    {
      autoView(vv, v, DeviceWrite);
      accelerator_for(i, sz, {	  
	  vv[i] = 0.31;
	});
    }

    ManagedArray<double> v2(11,0.1);
    v2 = v;
    
    assert(v.size() == sz);
    assert(v2.size() == sz);
        
    {
      autoView(vv, v2, HostRead);
      for(int i=0;i<sz;i++)
	assert(vv[i] == 0.31);
    }
  }

  
  assert(MemoryManager::globalPool().nOpenHandles() == 0);
  std::cout << "Tests passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testManagedArray();

  return 0;
}
