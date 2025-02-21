#pragma once
#include <Tensors.hpp>

//We implement activation functions as a numerical "mask" applied as an outer product to the output of the associated layer
//   for(int i=0;i<dim;i++)
//      for(int b=0;b<batch_size;b++)
//          out(i,b) *= mask(i,b)

class ReLU{
public: 
  inline Matrix operator()(const Matrix &x) const{
    int dim = x.size(0);
    int batch_size = x.size(1);
    Matrix out(dim,batch_size,1.0);
    //f(x)_i = max(x_i, 0)
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	if(x(i,b) <= 0.) out(i,b) = 0.;
    return out;
  }
};

class noActivation{
public:
  inline Matrix operator()(const Matrix &x) const{
    return Matrix(x.size(0),x.size(1),1.0);
  }
};
