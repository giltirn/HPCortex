#include <HPCortex.hpp>
#include <Testing.hpp>

typedef double FloatType; //more precise derivatives
typedef std::vector<FloatType> vecD;
typedef Tensor<FloatType,3> Tens;


void testSamePaddingZero1D(){
  FloatType delta = 1e-6;
  
  //Test SamePadding
  {
    //For a convolution of kernel size K and input size I, the output size is O=I-K+1
    //Same padding pads I->I' = I+K-1  such that O' = (I+K-1) - K + 1 = I
    int dims[3] = {1,6,1};
    Tens in(dims, vecD({1,2,3,4,5,6}) ); //(,1 2 3 4 5 6,)
    
    //K=1 I=6 I'=6+1-1
    {
      SamePaddingZero1D<FloatType> pd(1);      
      //(,1 2 3 4 5 6,) -> same
      Tens out = pd.padInput(in);
      assert(out.size(1) == 6);
      
      doHost(out, {
	  for(int i=0;i<6;i++)
	    assert(out_v(0,i,0) == FloatType(i+1));
	});

      Tens back = pd.unpadDeriv(out);
      assert(back.size(1) == 6);
      doHost(back, {
	  for(int i=0;i<6;i++)
	    assert(back_v(0,i,0) == FloatType(i+1));
	});
    }	    
    //K=3 I=6 I'=6+3-1
    {
      SamePaddingZero1D<FloatType> pd(3);      
      //(,1 2 3 4 5 6,) -> (, 0 1 2 3 4 5 6 0,)
      Tens out = pd.padInput(in);
      assert(out.size(1) == 8);

      vecD expect({0, 1, 2, 3, 4, 5, 6, 0});
      doHost(out, {
	  for(int i=0;i<8;i++)
	    assert(out_v(0,i,0) == expect[i]);
	});

      Tens back = pd.unpadDeriv(out);
      assert(back.size(1) == 6);
      doHost(back, {
	  for(int i=0;i<6;i++)
	    assert(back_v(0,i,0) == FloatType(i+1));
	});
    }	  
    //K=5 I=6 I'=6+5-1
    {
      SamePaddingZero1D<FloatType> pd(5);      
      //(,1 2 3 4 5 6,) -> (, 0 0 1 2 3 4 5 6 0 0,)
      Tens out = pd.padInput(in);
      assert(out.size(1) == 10);

      vecD expect({0, 0, 1, 2, 3, 4, 5, 6, 0, 0});
      doHost(out, {
	  for(int i=0;i<10;i++)
	    assert(out_v(0,i,0) == expect[i]);
	});

      Tens back = pd.unpadDeriv(out);
      assert(back.size(1) == 6);
      doHost(back, {
	  for(int i=0;i<6;i++)
	    assert(back_v(0,i,0) == FloatType(i+1));
	});
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

  static SamePaddingZero1D<FloatType> initPadding(int kernel_size){
    return SamePaddingZero1D<FloatType>(kernel_size);
  }
  
  static void testValue(const Tens &x, const Tens &init_filter, const Tens &got_value, const int out_chan, const int kernel_size){
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

      int in_data_len_pad = in_data_len  + kernel_size - 1;
      int edge_size = (kernel_size - 1)/2;

      for(int b=0;b<batch_size;b++){
	std::vector<vecD> channels_padded(in_chan, vecD(in_data_len_pad,0.) );
	for(int c=0;c<in_chan;c++)
	  for(int i=0;i<in_data_len;i++)
	    channels_padded[c][i+edge_size] = x_v(c,i,b);
     
	for(int out_chan_idx = 0 ; out_chan_idx < out_chan; out_chan_idx++){
	  for(int i=0; i < in_data_len_pad; i++){ //start of filter on padded input
	    int rem = in_data_len_pad - i;
	    if(rem >= kernel_size){
	      FloatType expect = 0;
	      for(int in_chan_idx=0;in_chan_idx<in_chan;in_chan_idx++){ //sum over channels	
		FloatType* data = channels_padded[in_chan_idx].data() + i;
		FloatType* ff = &f_v(out_chan_idx,in_chan_idx,0);
		
		expect += dot(ff, data, kernel_size);
	      }

	      if(expect <= 0.) expect = 0.; //ReLU
	      
	      FloatType got_dib = got_value_v(out_chan_idx,i,b);
	      
	      std::cout << out_chan_idx << " " << i << " (" << i-edge_size << ") " << b << " got " << got_dib << " expect " << expect << std::endl;
	      assert( abs_near( got_value_v(out_chan_idx,i,b), expect, 1e-8 ) );	  
	    }
	  }
	}
      }
    }
  }
};




template<>
struct convExpect<NoPadding<FloatType> >{

  static NoPadding<FloatType> initPadding(int kernel_size){
    return NoPadding<FloatType>();
  }
  
  static void testValue(const Tens &x, const Tens &init_filter, const Tens &got_value, const int out_chan, const int kernel_size){
    int in_chan = x.size(0);
    int in_data_len = x.size(1);
    int batch_size = x.size(2);

    int out_data_len = in_data_len - kernel_size + 1;
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
	  for(int i=0; i < in_data_len; i++){ 
	    int rem = in_data_len - i;
	    if(rem >= kernel_size){
	      FloatType expect = 0;
	      for(int in_chan_idx=0;in_chan_idx<in_chan;in_chan_idx++){ //sum over channels	
		FloatType* data = channels_data[in_chan_idx].data() + i;
		FloatType* ff = &f_v(out_chan_idx,in_chan_idx,0);
		
		expect += dot(ff, data, kernel_size);
	      }

	      if(expect <= 0.) expect = 0.; //ReLU
	      
	      FloatType got_dib = got_value_v(out_chan_idx,i,b);
	      
	      std::cout << out_chan_idx << " " << i << " " << b << " got " << got_dib << " expect " << expect << std::endl;
	      assert( abs_near( got_value_v(out_chan_idx,i,b), expect, 1e-8 ) );	  
	    }
	  }
	}
      }
    }
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

  int kernel_size =3;
  PaddingType pad = convExpect<PaddingType>::initPadding(kernel_size);
  
  int filter_dims[3] = { out_chan,in_chan,kernel_size };
  Tens init_filter(filter_dims);
  random(init_filter, rng);
  
  ReLU<FloatType> act;
    
  auto layer = conv1d_layer(input_layer<FloatType,Tens>(), init_filter, act, pad);

  int input_dims[3] = { in_chan, in_data_len, batch_size };
  Tens x(input_dims);
  random(x, rng);

  ////////////////////// TEST VALUE //////////////////////////////
  Tens got_value = layer.value(x);
  convExpect<PaddingType>::testValue(x, init_filter, got_value, out_chan, kernel_size);

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
    auto layer_tmp = conv1d_layer(input_layer<FloatType,Tens>(), init_filter, act, pad);    
    Tens new_filter(filter_dims);
    random(new_filter, rng);

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
    auto layer_tmp = conv1d_layer(input_layer<FloatType,Tens>(), init_filter, act, pad);
    Vector<FloatType> deriv(layer_tmp.nparams());
    random(deriv,rng);
    
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
	auto layer_tmp = conv1d_layer(input_layer<FloatType,Tens>(), pdelta, act, pad);	
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
	
  
  std::cout << "Conv1d tests passed" << std::endl;  
};

  


int main(int argc, char** argv){
  initialize(argc,argv);
  testConv1D<NoPadding<FloatType> >();
  testSamePaddingZero1D();
  testConv1D<SamePaddingZero1D<FloatType> >();
  return 0;
}
