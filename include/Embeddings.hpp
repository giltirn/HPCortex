#pragma once
#include <Tensors.hpp>

//For an input tensor of size C x E x B where C is the context window size, E is the embedding size and B is the batch size, embed the positions c \in C into the embeddings via the sin/cos method described in
//sec 3.5 of https://arxiv.org/pdf/1706.03762 and accessibly in https://www.geeksforgeeks.org/positional-encoding-in-transformers/
template<typename FloatType>
Tensor<FloatType,3> embedPositionsSinusoidal(const Tensor<FloatType,3> &in){
  Tensor<FloatType,3> out(in.sizeArray());
  int C = in.size(0), E = in.size(1), B = in.size(2);
  {
    autoView(out_v,out,DeviceWrite);
    autoView(in_v,in,DeviceRead);
    
    accelerator_for3d(b,B,e,E,c,C, 1,{
	size_t i = e/2;
	FloatType angle = FloatType(c)/pow(FloatType(10000.), FloatType(2*i)/E);
	out_v(c,e,b) = in_v(c,e,b) + (e % 2 == 0 ? sin(angle) : cos(angle));
    });
  }
  return out;
}
//Same as above but for unbatched data of size C x E
template<typename FloatType>
Tensor<FloatType,2> embedPositionsSinusoidal(const Tensor<FloatType,2> &in){
  Tensor<FloatType,2> out(in.sizeArray());
  int C = in.size(0), E = in.size(1);
  {
    autoView(out_v,out,DeviceWrite);
    autoView(in_v,in,DeviceRead);
    
    accelerator_for2d(e,E,c,C, 1,{
	size_t i = e/2;
	FloatType angle = FloatType(c)/pow(FloatType(10000.), FloatType(2*i)/E);
	out_v(c,e) = in_v(c,e) + (e % 2 == 0 ? sin(angle) : cos(angle));
    });
  }
  return out;
}
