#include <HPCortex.hpp>
#include <Testing.hpp>

typedef double FloatType; //more precise derivatives
typedef std::vector<FloatType> vecD;
typedef Tensor<FloatType,3> Tens;


void testSamePaddingZero1D(){
  std::vector<int> kernel_sizes = {1,2,3,4,5,6,7};
  std::vector<int> in_sizes = {3,4,9,10,19,20};
  std::vector<int> strides = {1,2,3,4};

  std::cout << "Testing same-padding" << std::endl;
  for(int stride: strides){
    for(int kernel_size: kernel_sizes){
      for(int in_size: in_sizes){
  
	int dims[3] = {1,in_size,1};
	vecD init(in_size);
	for(int i=0;i<in_size;i++)
	  init[i] = i+1;

	Tens in(dims, init );
	
	SamePaddingZero1D<FloatType> pd(kernel_size, stride);
	bool skip = false;
	Tens out;
	try{
	  out = pd.padInput(in);
	}catch(const std::exception &e){
	  std::cout << "stride=" << stride << " kernel_size=" << kernel_size << " in_size=" << in_size << " no symmetric solution" << std::endl;
	  skip=true;
	}
	if(skip) continue;
	  
	int in_size_padded = out.size(1);
	
	int conv_size = 0;
	for(int i=0;i<in_size_padded;i+=stride){
	  int rem = in_size_padded-i;
	  if(rem >= kernel_size) conv_size++;
	}

	assert(conv_size == in_size);

	assert( (in_size_padded - in_size) % 2 == 0 );
	int edge_size = (in_size_padded - in_size)/2;
	std::cout << "stride=" << stride << " kernel_size=" << kernel_size << " in_size=" << in_size << " in_size_padded=" << in_size_padded << " edge_size=" << edge_size << std::endl;
	
	doHost(out, {
	    for(int i=0;i<in_size_padded;i++){
	      //std::cout << i << ":" << out_v(0,i,0) << std::endl;
	      
	      if(i>= edge_size && i-edge_size < in_size)
		assert(out_v(0,i,0) == FloatType(i-edge_size+1));
	      else assert(out_v(0,i,0) == 0.0);
	    }	  	
	  });

	Tens back = pd.unpadDeriv(out);
	assert(back.size(1) == in_size);
	doHost(back, {
	    for(int i=0;i<in_size;i++)
	      assert(back_v(0,i,0) == FloatType(i+1));
	  });
      }
    }
  }

  std::cout << "Padding tests passed" << std::endl;
}

FloatType dot(FloatType const* a, FloatType const* b, int N){
  FloatType out = a[0]*b[0];
  for(int i=1;i<N;i++) out += a[i]*b[i];
  return out;  
}

template<typename PaddingType>
struct convExpect;

template<>
struct convExpect<SamePaddingZero1D<FloatType> >{

  static SamePaddingZero1D<FloatType> initPadding(int kernel_size, int stride){
    return SamePaddingZero1D<FloatType>(kernel_size,stride);
  }
  
  static void testValue(const Tens &x, const Tens &init_filter, const Tens &got_value, const int out_chan, const int kernel_size, const int stride){
    int in_chan = x.size(0);
    int in_data_len = x.size(1);
    int batch_size = x.size(2);

    int out_data_len = in_data_len; //same padding
    assert(got_value.size(0) == out_chan);
    assert(got_value.size(1) == out_data_len);
    assert(got_value.size(2) == batch_size);
    
    {
      autoView(x_v,x,HostRead);
      autoView(f_v,init_filter,HostRead);
      autoView(got_value_v,got_value,HostRead);

      int in_data_len_pad = stride*in_data_len + kernel_size - stride;
      if( (in_data_len_pad - in_data_len) % 2 != 0  && stride > 1){
	//we can add up to stride-1 without affecting the filtered output side. If the padding size is odd we need only add 1 to make it even
	in_data_len_pad += 1;
      }
      assert( (in_data_len_pad-in_data_len) % 2 == 0 );
      
      int edge_size = (in_data_len_pad-in_data_len)/2;

      for(int b=0;b<batch_size;b++){
	std::vector<vecD> channels_padded(in_chan, vecD(in_data_len_pad,0.) );
	for(int c=0;c<in_chan;c++)
	  for(int i=0;i<in_data_len;i++)
	    channels_padded[c][i+edge_size] = x_v(c,i,b);
     
	for(int out_chan_idx = 0 ; out_chan_idx < out_chan; out_chan_idx++){
	  int o = 0; //index in output
	  
	  for(int i=0; i < in_data_len_pad; i+= stride){ //start of filter on padded input
	    int rem = in_data_len_pad - i;
	    if(rem >= kernel_size){
	      FloatType expect = 0;
	      for(int in_chan_idx=0;in_chan_idx<in_chan;in_chan_idx++){ //sum over channels	
		FloatType* data = channels_padded[in_chan_idx].data() + i;
		FloatType* ff = &f_v(out_chan_idx,in_chan_idx,0);
		
		expect += dot(ff, data, kernel_size);
	      }

	      if(expect <= 0.) expect = 0.; //ReLU
	      
	      FloatType got_dob = got_value_v(out_chan_idx,o,b);
	      
	      std::cout << out_chan_idx << " " << i << " (" << i-edge_size << ") " << b << " got " << got_dob << " expect " << expect << std::endl;
	      assert( abs_near( got_dob, expect, 1e-8 ) );

	      ++o;
	    }
	  }
	  assert(o == out_data_len);
	} //out_chan
      } //batch idx
    } //scope
  }//test
};




template<>
struct convExpect<NoPadding<FloatType> >{

  static NoPadding<FloatType> initPadding(int kernel_size, int stride){
    return NoPadding<FloatType>();
  }
  
  static void testValue(const Tens &x, const Tens &init_filter, const Tens &got_value, const int out_chan, const int kernel_size, const int stride){
    int in_chan = x.size(0);
    int in_data_len = x.size(1);
    int batch_size = x.size(2);

    int out_data_len = (in_data_len - kernel_size + stride) / stride; //(I-K+S)//S

    int out_data_len_check = 0;
    for(int i=0;i<in_data_len;i+=stride){
      int rem = in_data_len - i;
      if(rem >= kernel_size)
	out_data_len_check++;
    }
    
    std::cout << "out_data_len got " << got_value.size(1) << " expect " << out_data_len << " expect check " << out_data_len_check << std::endl;
    assert(got_value.size(0) == out_chan);
    assert(got_value.size(1) == out_data_len);
    assert(got_value.size(2) == batch_size);
    
    {
      autoView(x_v,x,HostRead);
      autoView(f_v,init_filter,HostRead);
      autoView(got_value_v,got_value,HostRead);

      for(int b=0;b<batch_size;b++){
	std::vector<vecD> channels_data(in_chan, vecD(in_data_len) );
	for(int c=0;c<in_chan;c++)
	  for(int i=0;i<in_data_len;i++)
	    channels_data[c][i] = x_v(c,i,b);
     
	for(int out_chan_idx = 0 ; out_chan_idx < out_chan; out_chan_idx++){

	  int o = 0;
	  for(int i=0; i < in_data_len; i+=stride){ 
	    int rem = in_data_len - i;
	    if(rem >= kernel_size){
	      FloatType expect = 0;
	      for(int in_chan_idx=0;in_chan_idx<in_chan;in_chan_idx++){ //sum over channels	
		FloatType* data = channels_data[in_chan_idx].data() + i;
		FloatType* ff = &f_v(out_chan_idx,in_chan_idx,0);
		
		expect += dot(ff, data, kernel_size);
	      }

	      if(expect <= 0.) expect = 0.; //ReLU
	      
	      FloatType got_dob = got_value_v(out_chan_idx,o,b);
	      
	      std::cout << out_chan_idx << " " << i << " " << b << " got " << got_dob << " expect " << expect << std::endl;
	      assert( abs_near( got_dob, expect, 1e-8 ) );

	      o++;
	    }
	  }
	  assert(o == out_data_len);	  
	} //out chan
      } //batch idx
    }//scope
  }
};


  

template<typename PaddingType>
void testConv1D(){
  FloatType delta = 1e-6;
  std::mt19937 rng(1234);

  int in_chan = 3;
  int out_chan = 2;
  int batch_size = 3;
  int in_data_len = 8;
  
  std::vector<int> strides = {1,2,3};
  std::vector<int> kernel_sizes = {2,3,4};
  for(int stride : strides){
    for(int kernel_size : kernel_sizes){
      if(stride == 1 && kernel_size % 2 != 1) continue; //need symmetric edges
      
      std::cout << "Testing for stride=" << stride << " kernel_size=" << kernel_size << std::endl;

      PaddingType pad = convExpect<PaddingType>::initPadding(kernel_size, stride); 
      int filter_dims[3] = { out_chan,in_chan,kernel_size };
      Tens init_filter(filter_dims);
      uniformRandom(init_filter, rng);
  
      ReLU<FloatType> act;
    
      auto layer = conv1d_layer(input_layer<FloatType,Tens>(), init_filter, act, pad, stride);

      int input_dims[3] = { in_chan, in_data_len, batch_size };
      Tens x(input_dims);
      uniformRandom(x, rng);
      
      ////////////////////// TEST VALUE //////////////////////////////
      Tens got_value = layer.value(x);
      convExpect<PaddingType>::testValue(x, init_filter, got_value, out_chan, kernel_size, stride);
      
      int out_data_len = got_value.size(1);
 
      /////////////// TEST AUXILIARY FUNCTIONALITY /////////////////////
      assert(layer.nparams() == out_chan*in_chan*kernel_size);
      { //getParams
	Vector<FloatType> tp(layer.nparams());
	layer.getParams(tp,0);
	doHost2(tp,init_filter, {
	    int p=0;
	    for(int d=0;d<out_chan;d++)
	      for(int c=0;c<in_chan;c++)
		for(int k=0;k<kernel_size;k++){
		  assert(tp_v(p) == init_filter_v(d,c,k));
		  p++;
		}
	  });
      }
      { //update
	auto layer_tmp = conv1d_layer(input_layer<FloatType,Tens>(), init_filter, act, pad, stride);    
	Tens new_filter(filter_dims);
	uniformRandom(new_filter, rng);

	Vector<FloatType> new_filter_flat(layer_tmp.nparams());
	doHost2(new_filter_flat, new_filter, {
	    memcpy( new_filter_flat_v.data(), new_filter_v.data(), new_filter_flat.size(0)*sizeof(FloatType));
	  });
    
	layer_tmp.update(0, new_filter_flat);

	Vector<FloatType> tp(layer_tmp.nparams());
	layer_tmp.getParams(tp,0);
	doHost2(tp,new_filter, {
	    int p=0;
	    for(int d=0;d<out_chan;d++)
	      for(int c=0;c<in_chan;c++)
		for(int k=0;k<kernel_size;k++){
		  assert(tp_v(p) == new_filter_v(d,c,k));
		  p++;
		}
	  });
      }
      { //step
	auto layer_tmp = conv1d_layer(input_layer<FloatType,Tens>(), init_filter, act, pad, stride);
	Vector<FloatType> deriv(layer_tmp.nparams());
	uniformRandom(deriv,rng);
    
	layer_tmp.step(0, deriv, 1e-2);

	Vector<FloatType> tp(layer_tmp.nparams());
	layer_tmp.getParams(tp,0);
	doHost3(tp,init_filter,deriv, {
	    int p=0;
	    for(int d=0;d<out_chan;d++)
	      for(int c=0;c<in_chan;c++)
		for(int k=0;k<kernel_size;k++){
		  FloatType expect = init_filter_v(d,c,k) - deriv_v(p)*1e-2;	      
		  assert(fabs(tp_v(p) - expect) < 1e-9);
		  p++;
		}
	  });
      }

      ///////////////////////////// TEST DERIVS //////////////////////////////
      int out_sizes[3] = {out_chan, out_data_len, batch_size};
      Tens above_derivs(out_sizes,1.);
    
      Vector<FloatType> cost_deriv(layer.nparams(),0.);
      Tens in_deriv;
      layer.deriv(cost_deriv, 0, Tens(above_derivs), &in_deriv);

      //Cost deriv
      for(int d=0;d<out_chan;d++){
	for(int c=0;c<in_chan;c++){
	  for(int k=0;k<kernel_size;k++){
	    Tens pdelta = init_filter;
	    doHost(pdelta, { pdelta_v(d,c,k) += delta; });
	    auto layer_tmp = conv1d_layer(input_layer<FloatType,Tens>(), pdelta, act, pad, stride);	
	    Tens vdelta = layer_tmp.value(x);

	    int p = k+kernel_size*(c+in_chan*d);
	    FloatType cost_deriv_expect_p = 0.;
	
	    doHost2(vdelta,got_value, {
		for(int dd=0;dd<out_chan;dd++)
		  for(int o=0;o<out_data_len;o++)
		    for(int b=0;b<batch_size;b++)
		      cost_deriv_expect_p += (vdelta_v(dd,o,b) - got_value_v(dd,o,b))/delta;

	      });
	    doHost(cost_deriv, {	    
		std::cout << "Cost-deriv " << d << " "<< c << " " << k << " got " << cost_deriv_v(p) << "  expect " << cost_deriv_expect_p << std::endl;
		assert(abs_near( cost_deriv_v(p), cost_deriv_expect_p, 1e-4 ));
	      });
	  }
	}
      }
      //Input deriv
      assert(in_deriv.size(0) == in_chan && in_deriv.size(1) == in_data_len && in_deriv.size(2) == batch_size);
  
      for(int c=0;c<in_chan;c++){
	for(int i=0;i<in_data_len;i++){
	  for(int b=0;b<batch_size;b++){
	    Tens xdelta = x;
	    doHost(xdelta, { xdelta_v(c,i,b) += delta; });

	    Tens vdelta = layer.value(xdelta);
	
	    FloatType in_deriv_expect = 0.;
	
	    doHost2(vdelta,got_value, {
		for(int dd=0;dd<out_chan;dd++)
		  for(int o=0;o<out_data_len;o++)
		    for(int bb=0;bb<batch_size;bb++)
		      in_deriv_expect += (vdelta_v(dd,o,bb) - got_value_v(dd,o,bb))/delta;

	      });
	    doHost(in_deriv, {	    
		std::cout << "In-deriv " << c << " "<< i << " " << b << " got " << in_deriv_v(c,i,b) << "  expect " << in_deriv_expect << std::endl;
		assert(abs_near( in_deriv_v(c,i,b), in_deriv_expect, 1e-4 ));
	      });
	  }
	}
      }
    }
  }
  
  std::cout << "Conv1d tests passed" << std::endl;  
};


void testSamePaddingStride(){
  //For stride S>1 the output size = O=ceil( (I-K+1)/S ) = (I-K+1 +S-1)//S = (I-K+S)//S
  //for  I=(I'-K+S)//S     I'=SI+K-S      the padding amount is therefore SI+K-S-I which we require to be even

  std::vector<int> kernel_sizes = {1,3,5,7};
  std::vector<int> in_sizes = {3,4,9,10,19,20};
  std::vector<int> strides = {1,2,3,4};

  std::cout << "Test output size calculation" << std::endl;
  for(int stride: strides){
    for(int kernel_size: kernel_sizes){
      for(int in_size: in_sizes){
	
	int out_size = 0;
	for(int i=0;i<in_size;i+=stride){
	  int rem = in_size-i;
	  if(rem >= kernel_size) out_size++;
	}
	int out_size_expect = in_size >= kernel_size ? (in_size - kernel_size + stride)/stride : 0;

	std::cout << "stride=" << stride << " kernel_size="<< kernel_size << " in_size=" << in_size << " got " << out_size << " expect " << out_size_expect << std::endl;
	assert(out_size == out_size_expect);
      }
    }
  }

  std::cout << "Test same-padding calculation" << std::endl;
  for(int stride: strides){
    for(int kernel_size: kernel_sizes){
      for(int in_size: in_sizes){
	
	int in_size_pad = stride*in_size + kernel_size - stride;
	int out_size = 0;
	for(int i=0;i<in_size_pad;i+=stride){
	  int rem = in_size_pad-i;
	  if(rem >= kernel_size) out_size++;
	}

	std::cout << "stride=" << stride << " kernel_size="<< kernel_size << " in_size=" << in_size << " got " << out_size << " expect " << in_size << std::endl;
	assert(out_size == in_size);
      }
    }
  }
	
}
  
  

  
int main(int argc, char** argv){
  initialize(argc,argv);
  testSamePaddingStride();
  testSamePaddingZero1D();
  testConv1D<NoPadding<FloatType> >();
  testConv1D<SamePaddingZero1D<FloatType> >();
  
  return 0;
}
