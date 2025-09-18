#include <HPCortex.hpp>
#include <ManagedArray.hpp>
#include <Testing.hpp>

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


void testManagedTypeArray(){
  ManagedTypeArray< Tensor<double,2> > ar(2);
  
  std::vector<Tensor<double,2> > vr(2);
  vr[0] = Tensor<double,2>(2,3);
  vr[1] = Tensor<double,2>(1,4);  

  std::mt19937 rng(1234);
  uniformRandom(vr[0],rng);
  uniformRandom(vr[1],rng);

  ar[0] = vr[0];
  ar[1] = vr[1];

  {
    autoView(ar_v, ar, DeviceReadWrite);
    accelerator_for_gen(1,0,normal(), e, 2, {
	auto tv = ar_v[e];
	for(int i=0;i<tv.size(0);i++)
	  for(int j=0;j<tv.size(1);j++)
	    tv(i,j) = tv(i,j) + e+2*(j + tv.size(1)*i);
      });
  }
  {
    autoView(ar_v,ar, HostRead);
    for(int e=0;e<2;e++){
      autoView(tvo, vr[e], HostRead);
      auto tv = ar_v[e];
      for(int i=0;i<tv.size(0);i++)
	for(int j=0;j<tv.size(1);j++)
	  assert( abs_near(tv(i,j), tvo(i,j) + e+2*(j + tv.size(1)*i) , 1e-9 ) );
    }
  }

  //constructors
  {
    ManagedTypeArray< Tensor<double,2> > arcp(ar);
    for(int i=0;i<2;i++) assert(equal(arcp[i],ar[i]));

    ManagedTypeArray< Tensor<double,2> > armv(std::move(arcp));
    for(int i=0;i<2;i++) assert(equal(armv[i],ar[i]));
  }
  //assignment
  {
    ManagedTypeArray< Tensor<double,2> > arcp;
    arcp = ar;
    for(int i=0;i<2;i++) assert(equal(arcp[i],ar[i]));

    ManagedTypeArray< Tensor<double,2> > armv;
    armv = std::move(arcp);
    for(int i=0;i<2;i++) assert(equal(armv[i],ar[i]));
  }
  
  std::cout << "testManagedTypeArray passed" << std::endl;
}


int main(int argc, char** argv){
  initialize(argc,argv);
  
  testManagedArray();
  testManagedTypeArray();
  return 0;
}
