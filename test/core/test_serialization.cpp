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

  Endianness ends[2] = {Endianness::Little, Endianness::Big};
  
  //Test basic types
  {
    double v = 3.14;
    for(Endianness end : ends){
      {
	BinaryWriter wr("test.dat", end);
	wr.write(v);
      }
      double r;
      {
	BinaryReader rd("test.dat");
	rd.read(r);
      }
      assert(r==v);
    }
    std::pair<double,double> vp(6.455, 7.888);
    for(Endianness end : ends){
      {
	BinaryWriter wr("test.dat", end);
	wr.write(vp);
      }
      std::pair<double,double> r;
      {
	BinaryReader rd("test.dat");
	rd.read(r);
      }
      assert(r == vp);
    }
    std::vector<int> vv({3,4,6,7});
    for(Endianness end : ends){
      {
	BinaryWriter wr("test.dat", end);
	wr.write(vv);
      }
      std::vector<int> r;
      {
	BinaryReader rd("test.dat");
	rd.read(r);
      }
      assert(r.size() == vv.size());
      for(int i=0;i<r.size();i++)
	assert(r[i] == vv[i]);
    }
    std::vector<std::pair<float,float> > vvp({ {0.3,0.5}, {0.7,0.9} });
    for(Endianness end : ends){
      {
	BinaryWriter wr("test.dat", end);
	wr.write(vvp);
      }
      std::vector<std::pair<float,float> > r;
      {
	BinaryReader rd("test.dat");
	rd.read(r);
      }
      assert(r.size() == vvp.size());
      for(int i=0;i<r.size();i++)
	assert(r[i] == vvp[i]);
    }
    
  }
  
  //Test tensor io
  {
    int sz[3] = {5,6,7};
    Tensor<float,3> t(sz);
    uniformRandom(t,rng);


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
  }
  
  //Test model io
  {
    Matrix<float> w(5,6);
    Vector<float> b(5);
    uniformRandom(w,rng);
    uniformRandom(b,rng);  
    auto m = batch_tensor_dnn_layer<3>(w,b,0, ReLU<float>(), input_layer<confSingle,Tensor<float,3> >());
    auto mw = mse_cost(m);
  
    uniformRandom(w,rng);
    uniformRandom(b,rng);  
    auto mr = batch_tensor_dnn_layer<3>(w,b,0, ReLU<float>(), input_layer<confSingle,Tensor<float,3> >());
    auto mrw = mse_cost( batch_tensor_dnn_layer<3>(w,b,0, ReLU<float>(), input_layer<confSingle,Tensor<float,3> >()) );
  
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
  }
  
  std::cout << "testSerialization passed" << std::endl;
  
  return 0;
}
