#pragma once
#include<Buffers.hpp>

//Configuration class specifying common types for models
template<typename _FloatType, template <typename> class _BufferType>
struct ModelConfiguration{
  using FloatType = _FloatType;
  template<typename T>
  using BufferType = _BufferType<T>;
};
//single/double precision with no pipelining
using confSingle = ModelConfiguration<float, BufferSingle>;
using confDouble = ModelConfiguration<double, BufferSingle>;

//single/double precision with pipelining
using confSinglePipeline = ModelConfiguration<float, RingBuffer>;
using confDoublePipeline = ModelConfiguration<double, RingBuffer>;

using confSinglePipelineNew = ModelConfiguration<float, FillEmptyRingBuffer>;
using confDoublePipelineNew = ModelConfiguration<double, FillEmptyRingBuffer>;


#define EXTRACT_CONFIG_TYPES \
  typedef Config ModelConfig;			\
  typedef typename Config::FloatType FloatType;	\
  template<typename T> \
  using BufferType = typename Config::template BufferType<T>;
