#pragma once
#include "LayerCommon.hpp"
#include <layers/MultiHeadSelfAttentionLayer.hpp>
#include <layers/SkipConnection.hpp>
#include <layers/BatchTensorDNNlayer.hpp>

//A convenience wrapper for the transformer-decoder blocks described in eg https://arxiv.org/pdf/2305.07716
//Inputs are expected to be CxExB where C is the context size, E the embedding size and B the batch size
template<typename Below, typename ActivationFunc>
auto transformer_decoder_block(Below &&below,
			       int E, int nheads, int d_act, const ActivationFunc &activation);

#include "implementation/TransformerDecoderBlock.tcc"
