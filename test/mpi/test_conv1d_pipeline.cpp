#include <HPCortex.hpp>
#include <Testing.hpp>

template<typename ModelType, typename FloatType, typename InputType, typename OutputType>
void run(std::vector<FloatType> &loss,std::vector<Vector<FloatType> > &deriv, ModelType &model, const std::vector<InputType> &x, const std::vector<OutputType> &y){
  int ndata = x.size();
  auto cost = pipeline_mse_cost(model);
  int total_calls = ndata + cost.derivLag() -1; //after derivLag() calls we will get the first derivative so if ndata=1 we only need derivLag calls total

  int loss_off =0;
  int deriv_off = 0;
    
  for(int call=0;call < total_calls; call++){
    auto ls = cost.loss(x[call < ndata ? call : 0],y[call < ndata ? call : 0], DerivYes); //need to feed in extra vals to drain out the derivatives
    if(ls.second && loss_off < ndata)
      loss[loss_off++] = ls.first;
    auto der = cost.deriv();
    if(der.second){
      assert(deriv_off < ndata);
      deriv[deriv_off++] = der.first;
    }
  }
  std::cout << "deriv_off=" << deriv_off << " expect " << ndata << std::endl;
  assert(deriv_off == ndata);
}

void testConvPipeline(){
  typedef confSinglePipelineNew PipelineConfig;
  typedef confSingle StdConfig;
  typedef float FloatType;
  
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
  communicators().reportSetup();
  
  int nranks = communicators().pipelineNrank();
  assert(nranks > 2);
  int rank = communicators().pipelineRank();
  std::mt19937 rng(1234);

  typedef Tensor<FloatType,3> InputType;
  typedef Tensor<FloatType,3> OutputType;
  
  int in_channels = 2;
  int out_channels = 3;
  int in_len = 10;
  int out_len = 3;
  int kernel_size = 3;
  int stride = 1;

  int call_batch_size = 2;
  int batch_size = call_batch_size * 3 * nranks; //actual data batch size (must be divisible by call_batch_size)
  
  int output_data_sz[3] = { out_channels, out_len, batch_size };
  int input_data_sz[3] = {in_channels, in_len, batch_size};

  ///////////////////////////////////////////////////////////////////////
  //////// Generate data

  int ndata = 10;
  std::vector<OutputType> y(ndata, OutputType(output_data_sz)  );
  std::vector<InputType> x(ndata, InputType(input_data_sz)  );
  for(int i=0;i<ndata;i++){
    uniformRandom(x[i],rng);
    uniformRandom(y[i],rng);
  }
  
  ///////////////////////////////////////////////////////////////////////
  /////////////////////////// Convolution block /////////////////////////
  SamePaddingZero1D<FloatType> padding(kernel_size,stride);

  Tensor<FloatType,3> filter_init(out_channels, in_channels, kernel_size);
  uniformRandom(filter_init, rng);

  auto conv_block = conv1d_layer( filter_init, ReLU<FloatType>(), padding, stride, input_layer<PipelineConfig,InputType>() ); //last rank

  ///////////////////////////////////////////////////////////////////////
  ////// First DNN block
  int flat_size = in_len * out_channels;
  
  Matrix<FloatType> weight_init(flat_size, flat_size);
  uniformRandom(weight_init, rng);
  Vector<FloatType> bias_init(flat_size);
  uniformRandom(bias_init, rng);
  
  auto dnn_first = dnn_layer(weight_init, bias_init, ReLU<FloatType>(),
			     flatten_layer( input_layer<PipelineConfig,InputType>() )			      
			     ); //next to last rank


  ///////////////////////////////////////////////////////////////////////
  ////// Other DNN block
  auto dnn_other = dnn_layer(weight_init, bias_init, ReLU<FloatType>(),
			     input_layer<PipelineConfig,Matrix<FloatType> >()			      
			     );
  

  ///////////////////////////////////////////////////////////////////////
  /////// Output layer
  int out_flat_size = out_len * out_channels;
  
  Matrix<FloatType> weight_out_init(out_flat_size, flat_size);
  uniformRandom(weight_out_init, rng);
  Vector<FloatType> bias_out_init(out_flat_size);
  uniformRandom(bias_out_init, rng);

  int output_data_sz_block[3] = { out_channels, out_len, call_batch_size }; //within the pipeline blocks the batch size == call_batch_size
  auto output_block = unflatten_layer<3>(output_data_sz_block,
					 dnn_layer(weight_out_init, bias_out_init,
						   input_layer<PipelineConfig,Matrix<FloatType> >()						   
						   )
					 );
  
  static_assert(std::is_same< LAYEROUTPUTTYPE(decltype(output_block)), OutputType >::value );

  /////////////////////////////////////////////////////////////////////
  /////// Pipelined model
  auto model = pipeline_block_layer<OutputType>(call_batch_size, input_layer<PipelineConfig,InputType>());
  if(rank == 0) model.setRankBlock(conv_block);
  else if(rank == 1) model.setRankBlock(dnn_first);
  else if(rank != nranks-1) model.setRankBlock(dnn_other);
  else model.setRankBlock(output_block);

  auto cost = mse_cost( model );
  
  //////////////////////////////////////////////////////////////////////
  //Generate expectations
  auto full_model = enwrap( dnn_layer(weight_init, bias_init, ReLU<FloatType>(),
				      flatten_layer(
						    conv1d_layer(filter_init, ReLU<FloatType>(), padding, stride,
								 input_layer<StdConfig,InputType>()
								 ) 
						    )				      
				      )
			    );
  for(int r=2; r<nranks-1; r++){
    full_model = enwrap(dnn_layer(weight_init, bias_init, ReLU<FloatType>(),
				  std::move(full_model)				   
				  )
			);
  }
  auto full_model_complete = unflatten_layer<3>(output_data_sz,
						dnn_layer(weight_out_init, bias_out_init,
							  full_model							   
							   )						 
						);
  auto full_cost = mse_cost( full_model_complete );

  std::vector<FloatType> loss_expect(ndata);
  std::vector<Vector<FloatType> > deriv_expect(ndata);
  for(int d=0;d<ndata;d++){
    loss_expect[d] = full_cost.loss(x[d],y[d],DerivYes);
    deriv_expect[d] = full_cost.deriv();
  }
  
  ///////////////////////////////////////////////////////////////////////
  //Run
  std::vector<FloatType> loss(ndata);
  std::vector<Vector<FloatType> > deriv(ndata);
  for(int d=0;d<ndata;d++){
    loss[d] = cost.loss(x[d],y[d],DerivYes);
    deriv[d] = cost.deriv();
  }

  if(rank == 0){
    for(int i=0;i<ndata;i++){
      std::cout << "Loss " << i << " got " << loss[i] << " expect " << loss_expect[i] << std::endl;
      assert(abs_near(loss[i],loss_expect[i],FloatType(1e-4) ));
      assert(abs_near(deriv[i],deriv_expect[i],FloatType(1e-4),true));
    }
  }
  std::cout << "Tests passed" << std::endl;
}

int main(int argc, char** argv){
  initialize(argc, argv);
  
  testConvPipeline();

  return 0;
}

