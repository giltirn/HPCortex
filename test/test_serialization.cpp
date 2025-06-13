#include <HPCortex.hpp>
#include <Testing.hpp>

int main(int argc, char** argv){
  initialize(argc,argv);
  std::mt19937 rng(1234);
  
  //test endianness conversion
  if(endianness() == Endianness::Little){
    uint8_t vbig = 4;  //00000100
    uint8_t vlittle = bitReverse(vbig); //00100000    
    assert(vlittle == 32);
    std::cout << "System is little endian" << std::endl;
  }else{
    uint8_t vlittle = 4; //0010000
    uint8_t vbig = bitReverse(vlittle);  //00000100
    assert(vbig == 32);
    std::cout << "System is big endian" << std::endl;
  }

  //Test tensor io
  int sz[3] = {5,6,7};
  Tensor<float,3> t(sz);
  uniformRandom(t,rng);

  Endianness ends[2] = {Endianness::Little, Endianness::Big};
  for(Endianness end : ends){
    {
      BinaryWriter wr("test.dat", end);
      wr.write(t);
    }
    Tensor<float,3> tr(sz);
    {
      BinaryReader rd("test.dat");
      rd.read(tr);
    }
    assert(equal(tr,t));
  }     
  
  //Test model io
  Matrix<float> w(5,6);
  Vector<float> b(5);
  uniformRandom(w,rng);
  uniformRandom(b,rng);  
  auto m = batch_tensor_dnn_layer<3>(input_layer<float,Tensor<float,3> >(), w,b,0, ReLU<float>());
  auto mw = mse_cost(m);
  
  uniformRandom(w,rng);
  uniformRandom(b,rng);  
  auto mr = batch_tensor_dnn_layer<3>(input_layer<float,Tensor<float,3> >(), w,b,0, ReLU<float>());
  auto mrw = mse_cost( batch_tensor_dnn_layer<3>(input_layer<float,Tensor<float,3> >(), w,b,0, ReLU<float>()) );
  
  for(Endianness end : ends){
    {
      BinaryWriter wr("test.dat", end);
      wr.write(m);
    }
    {
      BinaryReader rd("test.dat");
      rd.read(mr);
    }
    Vector<float> pm(m.nparams());
    m.getParams(pm,0);
    Vector<float> pmr(m.nparams());
    mr.getParams(pmr,0);
        
    assert(equal(pm,pmr));
  }     

  for(Endianness end : ends){
    {
      BinaryWriter wr("test2.dat", end);
      wr.write(mw);
    }
    {
      BinaryReader rd("test2.dat");
      rd.read(mrw);
    }
    Vector<float> pm = mw.getParams();
    Vector<float> pmr = mrw.getParams();
        
    assert(equal(pm,pmr));
  }   
  
  return 0;
}
