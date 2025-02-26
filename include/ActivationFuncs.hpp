#pragma once
#include <Tensors.hpp>

//We implement activation functions as a numerical "mask" applied as an outer product to the output of the associated layer
//   for(int i=0;i<dim;i++)
//      for(int b=0;b<batch_size;b++)
//          out(i,b) *= mask(i,b)

class ReLU{
public: 
  Matrix operator()(const Matrix &x) const;
};

class noActivation{
public:
  inline Matrix operator()(const Matrix &x) const{
    return Matrix(x.size(0),x.size(1),1.0);
  }
};
