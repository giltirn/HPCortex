#pragma once
#include <Tensors.hpp>

template<typename FloatType>
class NoPadding{
public:
  template<int Dim>
  inline Tensor<FloatType,Dim> padInput(const Tensor<FloatType,Dim> &in) const{ return in; }
  template<int Dim>
  inline Tensor<FloatType, 3> unpadDeriv(const Tensor<FloatType, Dim> &deriv_pad) const{ return deriv_pad; } 
};

//For a convolution of kernel size K and input size I, the output size is O=I-K+1 with stride=1 
//Same padding pads I->I' = I+K-1  such that O' = (I+K-1) - K + 1 = I
//We will assume K is odd so that the K-1 padding sites are symmetric

//For stride S>1 the output size = O=ceil( (I-K+1)/S ) = (I-K+1 +S-1)//S = (I-K+S)//S
//for  I=(I'-K+S)//S     I'=SI+K-S      the padding amount is therefore SI+K-S-I which we require to be even

template<typename FloatType>
class SamePaddingZero1D{
  int kernel_size;
  int stride;
public:
  SamePaddingZero1D(int kernel_size, int stride=1): kernel_size(kernel_size), stride(stride){ }

  Tensor<FloatType, 3> padInput(const Tensor<FloatType, 3> &in) const{
    int channels = in.size(0);
    int batch_size = in.size(2);
    int in_size = in.size(1);

    int out_size = stride*in_size + kernel_size - stride;
    if( (out_size - in_size) % 2 != 0  && stride > 1){
      //we can add up to stride-1 without affecting the filtered output side. If the padding size is odd we need only add 1 to make it even
      out_size += 1;
    }

    if( (out_size - in_size) % 2 != 0 ) throw std::runtime_error("Require padding to be symmetric");
    
    int edge_size = (out_size - in_size)/2;

    int dims[3] = {channels, out_size, batch_size};
    Tensor<FloatType, 3> out(dims);
    autoView(out_v,out,DeviceWrite);
    autoView(in_v,in,DeviceRead);
    
    accelerator_for3d(b,batch_size,o,out_size,c,channels,  1,{
	int i = o-edge_size;
	if(i<0 || i>=in_size) out_v(c,o,b) = 0.;
	else out_v(c,o,b) = in_v(c,i,b);
      });
    return out;
  }
  //dcost/din_{ci} = \sum_j dcost/dout_{cj} dout_{cj}/din_{ci}
  //out_cj = j >= edge_size && j<edge_size+data_len ? in_{c,j-edge_size} : 0
  //dout_cj /din_ci = j>= edge_size && j<edge_size+data_len ? delta_{j-edge_size,i} : 0
  //dcost/din_{ci} = dcost/dout_{c,i+edge_size}  
  Tensor<FloatType, 3> unpadDeriv(const Tensor<FloatType, 3> &deriv_pad) const{
    int channels = deriv_pad.size(0);
    int batch_size = deriv_pad.size(2);
    int padded_len = deriv_pad.size(1);

    //padded_len = stride*unpadded-len + kernel_size - stride;
    int unpadded_len  = (padded_len - kernel_size + stride)/stride;

    int edge_size = (padded_len - unpadded_len)/2;

    int dims[3] = {channels, unpadded_len, batch_size};
    Tensor<FloatType, 3> out(dims);
    autoView(out_v,out,DeviceWrite);
    autoView(deriv_pad_v,deriv_pad,DeviceRead);
    
    accelerator_for3d(b,batch_size,i,unpadded_len,c,channels,  1,{
	int o = i+edge_size;
	out_v(c,i,b) = deriv_pad_v(c,o,b);
      });
    return out;
  }
 

  
  
};
