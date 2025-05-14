#include <HPCortex.hpp>
#include <Testing.hpp>

void testBatching(){
  typedef double FloatType;
  
  typedef std::vector<FloatType> vec;
  
  typedef Tensor<FloatType,2> Dtype;
  typedef Tensor<FloatType,3> Btype;

  int ndata = 20;
  
  std::vector<int> order(ndata);
  for(int i=0;i<ndata;i++) order[i] = i;
  std::default_random_engine gen(5684);
  std::uniform_int_distribution<int> dist(0,ndata-1);
  std::random_shuffle ( order.begin(), order.end(), [&](const int l){ return dist(gen); }  );

  int dsizex[2] = {2,3};
  int dsizey[2] = {1,2};
  
  std::vector<XYpair<FloatType, 2,2> > data(ndata);
  for(int i=0;i<ndata;i++){
    Dtype x(dsizex);    
    random(x,gen);
    Dtype y(dsizey);
    random(y,gen);
    data[i].x = std::move(x);
    data[i].y = std::move(y);
  }
  int batch_size = 3;
  
  XYpair<FloatType, 3,3> bdata = batchData(order.data(), batch_size, data);
  std::cout << "Got x size " << bdata.x.sizeArrayString() << " y size " << bdata.y.sizeArrayString() << std::endl;
  
  assert(bdata.x.size(2) == batch_size);
  assert(bdata.y.size(2) == batch_size);
  for(int d=0;d<2;d++){
    assert(bdata.x.size(d) == dsizex[d]);
    assert(bdata.y.size(d) == dsizey[d]);
  }
  
  for(int b=0;b<batch_size;b++){
    Dtype gotx = bdata.x.peekLastDimension(b);
    assert(equal(gotx, data[order[b]].x));

    Dtype goty = bdata.y.peekLastDimension(b);
    assert(equal(goty, data[order[b]].y));
  }
  std::cout << "Test passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc,argv);
  
  testBatching();

  return 0;
}
