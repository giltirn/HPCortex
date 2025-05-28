#pragma once
#include "LayerCommon.hpp"
#include <layers/MultiHeadSelfAttentionLayer.hpp>
#include <layers/MultiHeadCrossAttentionLayer.hpp>
#include <layers/SkipConnection.hpp>
#include <layers/BatchTensorDNNlayer.hpp>

//A convenience wrapper for the transformer-decoder blocks described in eg https://arxiv.org/pdf/2305.07716
//Inputs are expected to be CxExB where C is the context size, E the embedding size and B the batch size
template<typename Below, typename ActivationFunc>
auto transformer_decoder_block(Below &&below,
			       int E, int nheads, int d_act, const ActivationFunc &activation);

template<typename Below, typename ActivationFunc>
auto transformer_encoder_block(Below &&below,
			       int E, int nheads, int d_act, const ActivationFunc &activation);

//The cross-attention decoder block a la https://arxiv.org/pdf/1706.03762 but using pre-LN normalization
template<typename EncoderInput, typename DecoderInput, typename ActivationFunc>
auto transformer_cross_decoder_block(EncoderInput &&encoder_in, DecoderInput &&decoder_in,
				     int E, int nheads, int d_act , const ActivationFunc &activation);

#include "implementation/TransformerEncoderDecoderBlock.tcc"
