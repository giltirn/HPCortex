#include <HPCortex.hpp>
#include <Testing.hpp>

void testTransformBatchMatrix(){
  std::mt19937 rng(1234);
  typedef double FloatType;
  int sz[3] = {4,5,6};
  Tensor<FloatType,3> tens(sz);
  uniformRandom(tens,rng);

  {
    int mat_size = sz[0]*sz[1];
    Vector<FloatType> got = transformBatchMatrix(0,1,tens);
    {
      autoView(tens_v,tens,HostRead);
      autoView(got_v,got,HostRead);
    
      for(int o=0;o<sz[2];o++){
	int off = o*mat_size;
	FloatType const* mat = got_v.data() + off;
	for(int r=0;r<sz[0];r++)
	  for(int c=0;c<sz[1];c++)
	    assert(mat[c + sz[1]*r] == tens_v(r,c,o));
      }
    }
    Tensor<FloatType,3> bak(sz);
    untransformBatchMatrix(0,1,bak, got);
    assert(equal(bak,tens,true));
  }

  {
    int mat_size = sz[0]*sz[1];
    Vector<FloatType> got = transformBatchMatrix(1,0,tens);
    {
      autoView(tens_v,tens,HostRead);
      autoView(got_v,got,HostRead);
      for(int o=0;o<sz[2];o++){
	int off = o*mat_size;
	FloatType const* mat = got_v.data() + off;
	for(int r=0;r<sz[1];r++)
	  for(int c=0;c<sz[0];c++)
	    assert(mat[c + sz[0]*r] == tens_v(c,r,o));
      }
    }
    Tensor<FloatType,3> bak(sz);
    untransformBatchMatrix(1,0,bak, got);
    assert(equal(bak,tens,true));
  }

  int sz4[4] = {3,4,5,6};
  Tensor<FloatType,4> tens4(sz4);
  uniformRandom(tens4,rng);

  {
    int rowdim = 2;
    int coldim = 1;
    int rows = sz4[rowdim], cols= sz4[coldim];
    int other_dim1 = 0, other_dim2 = 3;

    int mat_size = rows*cols;
    Vector<FloatType> got = transformBatchMatrix(rowdim, coldim, tens4);

    {
      autoView(tens4_v,tens4,HostRead);
      autoView(got_v,got,HostRead);
      
      for(int o=0;o<sz4[other_dim1]*sz4[other_dim2];o++){
	int off = o*mat_size;
	FloatType const* mat = got_v.data() + off;
	
	int i = o / sz4[other_dim2];
	int j = o % sz4[other_dim2];
      
	for(int r=0;r<rows;r++)
	  for(int c=0;c<cols;c++)
	    assert(mat[c + cols*r] == tens4_v(i,c,r,j));
      }
    }
    Tensor<FloatType,4> bak(sz4);
    untransformBatchMatrix(rowdim, coldim, bak, got);
    assert(equal(bak,tens4,true));
  }
  std::cout << "testTransformBatchMatrix passed" << std::endl;
}

void testTransformBatchVector(){
  std::mt19937 rng(1234);
  typedef double FloatType;

  std::vector<int> batch_sizes = {1,4,5,8,32,64};
  std::vector<int> vec_sizes = { 2,3,4,5,8,32,64,128,256 };
  std::vector<int> other_sizes = {1,2,4,5,8,32,64};

  for(int vecsz : vec_sizes){
    for(int other_size : other_sizes){
      for(int batch_size : batch_sizes){
	std::cout << "vec_size: " << vecsz << " other_size: " << other_size << " batch_size: " << batch_size << std::endl;

	{
	  int sz[3] = {vecsz,other_size,batch_size};
	  Tensor<FloatType,3> tens(sz);
	  uniformRandom(tens,rng);
	
	  Vector<FloatType> got = transformBatchVector(0,tens);
	  {
	    autoView(tens_v,tens,HostRead);
	    autoView(got_v,got,HostRead);
	  
	    for(int o=0;o<sz[1];o++){
	      for(int b=0;b<sz[2];b++){
		int off = (b +sz[2]*o)*vecsz;
		FloatType const* vec = got_v.data() + off;
		for(int r=0;r<sz[0];r++)
		  assert(vec[r] == tens_v(r,o,b));
	      }
	    }
	  }
	  Tensor<FloatType,3> bak(sz);
	  untransformBatchVector(0,bak, got);
	  assert(equal(bak,tens,true));
	}
      
	{
	  int sz[3] = {other_size,vecsz,batch_size};
	  Tensor<FloatType,3> tens(sz);
	  uniformRandom(tens,rng);
	
	  Vector<FloatType> got = transformBatchVector(1,tens);
	  {
	    autoView(tens_v,tens,HostRead);
	    autoView(got_v,got,HostRead);
      
	    for(int o=0;o<sz[0];o++){
	      for(int b=0;b<sz[2];b++){
		int off = (b +sz[2]*o)*vecsz;
		FloatType const* vec = got_v.data() + off;
		for(int r=0;r<sz[1];r++)
		  assert(vec[r] == tens_v(o,r,b));
	      }
	    }
      
	    Tensor<FloatType,3> bak(sz);
	    untransformBatchVector(1,bak, got);
	    assert(equal(bak,tens,true));
	  }
	}


	{
	  int sz[4] = {vecsz,other_size,other_size,batch_size};
	  Tensor<FloatType,4> tens(sz);
	  uniformRandom(tens,rng);
	
	  Vector<FloatType> got = transformBatchVector(0,tens);
	  {
	    autoView(tens_v,tens,HostRead);
	    autoView(got_v,got,HostRead);
	  
	    for(int o=0;o<sz[1];o++){
	      for(int p=0;p<sz[2];p++){		
		for(int b=0;b<sz[3];b++){
		  int off = (b +sz[3]*(p + sz[2]*o))*vecsz;
		  FloatType const* vec = got_v.data() + off;
		  for(int r=0;r<sz[0];r++)
		    assert(vec[r] == tens_v(r,o,p,b));
		}
	      }
	    }
	  }
	  Tensor<FloatType,4> bak(sz);
	  untransformBatchVector(0,bak, got);
	  assert(equal(bak,tens,true));
	}


	
      }
    }
  }
  std::cout << "testTransformBatchVector passed" << std::endl;
}


int main(int argc, char** argv){
  initialize(argc,argv);
  testTransformBatchMatrix();
  testTransformBatchVector();
  return 0;
}
