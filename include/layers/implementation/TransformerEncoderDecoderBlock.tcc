namespace TransformerEncoderDecoderBlock{
  constexpr int embedding_dim = 1;

  //These functions build the layers from the bottom up

  //applying the layer-norm just below the multi-head attention (as opposed to after the skip block) is referred to as a pre-LN setup, and apparently has beneficial convergence properties (https://arxiv.org/pdf/2002.04745)
  template<typename Below>
  auto declare_skip_block_1(Below &&below,			    
			    int E,
			    int nheads,  //define  d_qkv = E/nheads     E must divide exactly by nheads
			    bool use_mask //encoder: false, decoder: true
			    ){
    typedef FLOATTYPE(Below) FloatType;
    typedef Tensor<FloatType,3> LayerInputType;

    return skip_connection(
			   multihead_self_attention_layer(
							  norm_layer<3>(
									input_layer<FloatType, LayerInputType>(),
									embedding_dim, E, true, true
									),
							  nheads, E, use_mask
							  ),
			   std::move(below)
			   );
  }  

  template<typename Below>
  auto declare_block_connection(int E,
				Below && below){
    return norm_layer<3>(
			 std::move(below),
			 embedding_dim, E, true, true
			 );
  }
  
  template<typename Below, typename ActivationFunc>
  auto declare_skip_block_2(int E,
			    int d_act, //neurons in hidden layer			    
			    const ActivationFunc &activation,
			    Below && below){
    typedef FLOATTYPE(Below) FloatType;
    typedef Tensor<FloatType,3> LayerInputType;
    return skip_connection(
			   batch_tensor_dnn_layer<3>( //linear layer with no activation, size E x d_act
						     batch_tensor_dnn_layer<3>(
									       input_layer<FloatType,LayerInputType>(),
									       embedding_dim, d_act, E, activation
									       ), //linear layer with activation size d_act x E
						     embedding_dim, E, d_act, noActivation<FloatType>()
						     ),
			   std::move(below)
			   );
  }

  template<typename Below, typename ActivationFunc>
  auto transformer_encoder_decoder_block(Below &&below,
					 int E, int nheads, int d_act, const ActivationFunc &activation, bool use_mask){
    auto block1 = declare_skip_block_1(std::forward<Below>(below),E,nheads, use_mask);
    auto connection = declare_block_connection(E,std::move(block1));
    auto decoder = declare_skip_block_2(E, d_act, activation, std::move(connection));
    return decoder;
  }

  //we will use pre-LN normalization here also, at least for the value side
  template<typename EncoderInput, typename DecoderInput>
  auto declare_cross_attention_block(EncoderInput &&encoder_in, DecoderInput &&decoder_in,
				     int E, int nheads){
    typedef FLOATTYPE(EncoderInput) FloatType;
    typedef Tensor<FloatType,3> LayerInputType;

    auto repl = replicate_layer(std::move(decoder_in),2);
    auto xlayer = multihead_cross_attention_layer(std::forward<EncoderInput>(encoder_in),
						  norm_layer<3>(
								std::move(*repl[0]),
								embedding_dim, E, true, true
								),
						  nheads, E, true //always use mask
						  );
    return sum_join_layer( std::move(xlayer),
			   std::move(*repl[1]) );
  }

  template<typename EncoderInput, typename DecoderInput, typename ActivationFunc>
  auto transformer_cross_decoder_block(EncoderInput &&encoder_in, DecoderInput &&decoder_in,
				       int E, int nheads, int d_act , const ActivationFunc &activation){
    auto block1 = declare_skip_block_1(std::forward<DecoderInput>(decoder_in),E,nheads, true); //use mask
    auto block2 = declare_cross_attention_block(std::forward<EncoderInput>(encoder_in), std::move(block1), E, nheads);
    auto connection = declare_block_connection(E,std::move(block2));
    auto decoder = declare_skip_block_2(E, d_act, activation, std::move(connection));
    return decoder;
  }
  
};


template<typename Below, typename ActivationFunc>
auto transformer_decoder_block(Below &&below,
			       int E, int nheads, int d_act, const ActivationFunc &activation){
  return TransformerEncoderDecoderBlock::transformer_encoder_decoder_block(std::forward<Below>(below),E,nheads,d_act,activation,true);
}
template<typename Below, typename ActivationFunc>
auto transformer_encoder_block(Below &&below,
			       int E, int nheads, int d_act, const ActivationFunc &activation){
  return TransformerEncoderDecoderBlock::transformer_encoder_decoder_block(std::forward<Below>(below),E,nheads,d_act,activation,false);
}
template<typename EncoderInput, typename DecoderInput, typename ActivationFunc>
auto transformer_cross_decoder_block(EncoderInput &&encoder_in, DecoderInput &&decoder_in,
				     int E, int nheads, int d_act , const ActivationFunc &activation){
  return TransformerEncoderDecoderBlock::transformer_cross_decoder_block(std::forward<EncoderInput>(encoder_in), std::forward<DecoderInput>(decoder_in), E,nheads,d_act,activation);
}
