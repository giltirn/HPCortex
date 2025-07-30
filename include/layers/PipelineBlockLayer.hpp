#pragma once
#include "LayerCommon.hpp"
#include <LayerIOcontainer.hpp>
/**
 * @brief Base class for pipeline layer blocks
 */
template<typename FloatType>
class PipelineBlockContainerBase{
public:
  /**
   * @brief Evaluate the block with wrapped inputs/outputs
   */ 
  virtual LayerIOcontainer blockValue(const LayerIOcontainer &block_input, EnableDeriv enable_deriv) = 0;
  /**
   * @brief Evaluate the derivative with wrapped inputs/outputs
   */ 
  virtual void blockDeriv(Vector<FloatType> &cost_deriv, const LayerIOcontainer &_above_deriv, LayerIOcontainer &layer_input_deriv) = 0;
  /**
   * @brief Update the block parameters
   */ 
  virtual void blockUpdate(int off, const Vector<FloatType> &new_params) = 0;
  /**
   * @brief Update the block parameters using derivative and step size
   */ 
  virtual void blockStep(int off, const Vector<FloatType> &derivs, FloatType eps) = 0;
  /**
   * @brief Get the block parameters
   */ 
  virtual void blockGetParams(Vector<FloatType> &rank_params) = 0;
  /**
   * @brief Get the count of block floating point ops
   */ 
  virtual size_t blockFLOPS(int value_or_deriv) const = 0;

  virtual void resizeInputBuffer(int to) = 0;

  /**
   * @brief Set the underlying type of 'con' to the block's input type
   */   
  virtual void setInputType(LayerIOcontainer &con) const = 0;

  /**
   * @brief Set the underlying type of 'con' to the block's output type
   */   
  virtual void setOutputType(LayerIOcontainer &con) const = 0;
  
  virtual int nparams() const = 0;

  virtual ~PipelineBlockContainerBase(){}
};

/**
 * @brief Implementation of the above for a generic block instance store object
 */   
template<typename BlockStore>
class PipelineBlockContainer: public PipelineBlockContainerBase<typename BlockStore::type::FloatType>{
  BlockStore block;
  typedef typename BlockStore::type BlockType;
  typedef typename BlockType::InputType BlockInputType;
  typedef typename BlockType::FloatType FloatType;
  typedef LAYEROUTPUTTYPE(BlockType) BlockOutputType;
public:
  PipelineBlockContainer(BlockStore &&block): block(std::move(block)){}
  
  LayerIOcontainer blockValue(const LayerIOcontainer &block_input, EnableDeriv enable_deriv) override{
    const BlockInputType &block_input_tens = block_input.as<BlockInputType>();
    return LayerIOcontainer(block.v.value(block_input_tens,enable_deriv));
  }
  void blockDeriv(Vector<FloatType> &cost_deriv, const LayerIOcontainer &_above_deriv, LayerIOcontainer &_layer_input_deriv) override{
    BlockOutputType above_deriv(_above_deriv.as<BlockOutputType>());
    BlockInputType layer_input_deriv_tmp;
    block.v.deriv(cost_deriv, 0, std::move(above_deriv), &layer_input_deriv_tmp);
    _layer_input_deriv = std::move(layer_input_deriv_tmp);
  }
  void blockUpdate(int off, const Vector<FloatType> &new_params) override{
    block.v.update(off,new_params);
  }
  void blockStep(int off, const Vector<FloatType> &derivs, FloatType eps) override{
    block.v.step(off, derivs, eps);
  }
  void blockGetParams(Vector<FloatType> &rank_params) override{
    block.v.getParams(rank_params, 0);
  }
  void resizeInputBuffer(int to) override{
    block.v.resizeInputBuffer(to);
  }
  size_t blockFLOPS(int value_or_deriv) const override{
    return block.v.FLOPS(value_or_deriv);
  }    
  void setInputType(LayerIOcontainer &con) const override{
    con.setType<BlockInputType>();
  }
  void setOutputType(LayerIOcontainer &con) const override{
    con.setType<BlockOutputType>();
  }
  int nparams() const override{
    return block.v.nparams();
  }
};

//Note: the pipeline starts with rank 0!
template<typename LayerOutputType, typename BelowStore>
class PipelineBlockLayer{
public:
  typedef typename BelowStore::type BelowType;
  typedef typename BelowType::ModelConfig Config;
  EXTRACT_CONFIG_TYPES;
  typedef LAYEROUTPUTTYPE(BelowType) LayerInputType;
  typedef typename BelowType::InputType InputType;
  
private: 
  BelowStore below;
  
  int ubatch_size;
  std::unique_ptr<PipelineBlockContainerBase<FloatType> > rank_block;
  bool initialized;

  int rank;
  int pipeline_depth;
  bool is_first;
  bool is_last;

  typedef std::unique_ptr<LayerIOcontainerInitializer> initStore;
  initStore rank_block_in_init;
  initStore pipeline_out_init;

  mutable initStore rank_above_deriv_init;
  mutable initStore pipeline_input_deriv_init;
  
  int nparam;
  std::vector<int> rank_block_nparam;
  int rank_param_offset;

  size_t value_flops;
  mutable size_t deriv_flops;

  /**
   * @brief Initialize the block layer
   */
  void initialize();
  /**
   * @brief Gather a parameter vector to rank 0
   */
  void gatherParameterVector(int to_off, Vector<FloatType> &vec_to, Vector<FloatType> &vec_from) const;
  
public:
  typedef LeafTag tag;
  
  PipelineBlockLayer(BelowStore &&below, int ubatch_size): below(std::move(below)), ubatch_size(ubatch_size), initialized(false), nparam(0), rank(communicators().pipelineRank()),
							   pipeline_depth(communicators().pipelineNrank()), is_first(rank == 0), is_last(rank == pipeline_depth -1), 
							   rank_block_nparam(communicators().pipelineNrank(),0){}

  /**
   * @brief Set the block for the current rank
   */
  template<typename Block>
  void setRankBlock(Block &&block){
    if(initialized) throw std::runtime_error("Cannot change model once initialized");
    rank_block.reset(new PipelineBlockContainer<DDST(block)>(std::forward<Block>(block)));
  }

  int nparams(){
    initialize();
    return nparam + below.v.nparams();
  }
  
  LayerOutputType value(const InputType &x, EnableDeriv enable_deriv = DerivNo);
  
  int deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&_above_deriv, InputType* input_above_deriv_return = nullptr) const;

  inline void resizeInputBuffer(size_t to);
  
  int update(int off, const Vector<FloatType> &new_params);
  
  int step(int off, const Vector<FloatType> &derivs, FloatType eps);
  
  int getParams(Vector<FloatType> &into, int off) const;
  
  size_t FLOPS(int value_or_deriv) const;   
};

/**
 * @brief Create a pipeline block layer object
 * @param ubatch_size The microbatch size. Must be a divisor of the global batch size
 * @param below The layer below
 */
template<typename LayerOutputType, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto pipeline_block_layer(int ubatch_size, U &&below){
  return PipelineBlockLayer<LayerOutputType, DDST(below)>(std::forward<U>(below), ubatch_size);
}

#include "implementation/PipelineBlockLayer.tcc"
