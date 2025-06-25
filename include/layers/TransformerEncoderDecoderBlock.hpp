#pragma once
#include "LayerCommon.hpp"
#include <layers/MultiHeadSelfAttentionLayer.hpp>
#include <layers/MultiHeadCrossAttentionLayer.hpp>
#include <layers/SkipConnection.hpp>
#include <layers/BatchTensorDNNlayer.hpp>

//A convenience wrapper for the transformer-decoder blocks described in eg https://arxiv.org/pdf/2305.07716
//Inputs are expected to be CxExB where C is the context size, E the embedding size and B the batch size
template<typename Below, typename ActivationFunc>
auto transformer_decoder_block(int E, int nheads, int d_act, const ActivationFunc &activation,
			       Below &&below);

template<typename Below, typename ActivationFunc>
auto transformer_encoder_block(int E, int nheads, int d_act, const ActivationFunc &activation,
			       Below &&below);

//The cross-attention decoder block a la https://arxiv.org/pdf/1706.03762 but using pre-LN normalization
template<typename EncoderInput, typename DecoderInput, typename ActivationFunc>
auto transformer_cross_decoder_block(int E, int nheads, int d_act , const ActivationFunc &activation,
				     EncoderInput &&encoder_in, DecoderInput &&decoder_in);

#include "implementation/TransformerEncoderDecoderBlock.tcc"
