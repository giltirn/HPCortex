#pragma once
#include<Tensors.hpp>
#include<Layers.hpp>
#include<Comms.hpp>

//This is a pipeline parallelized cost function wrapper. The user must provide a PipelineBlock containing the layers on the current rank, each terminating
//with an InputLayer.
//call_batch_size must be provided indicating the batch size acted on by each rank in each cycle; it must be a divisor of the total batch size
template<typename PipelineBlockType, typename CostFunc>
class BatchPipelineCostFuncWrapper{
public:
  typedef typename PipelineBlockType::FloatType FloatType;
  typedef typename PipelineBlockType::InputType InputType; //overall input type
  typedef typename PipelineBlockType::OutputType OutputType; //overall output type
  static_assert( std::is_same<OutputType, typename CostFunc::DataType>::value == true );
  
  //types for this rank's block
  typedef typename PipelineBlockType::BlockInputType BlockInputType;
  typedef LAYEROUTPUTTYPE(PipelineBlockType) BlockOutputType; 

  enum { InputDimension = InputType::Dimension, OutputDimension = OutputType::Dimension };
private:  
  PipelineBlockType &block;
  
  CostFunc cost;
  int nparam;
  int value_lag;
  int deriv_lag;

  int call_batch_size;
  int rank;
  int nrank;
  
  Vector<FloatType> deriv_store;
public:
  BatchPipelineCostFuncWrapper(PipelineBlockType &block, int call_batch_size, const CostFunc &cost = CostFunc()): cost(cost), block(block), nparam(block.nparams()),
														  value_lag(block.valueLag()), deriv_lag(block.derivLag()),
														  call_batch_size(call_batch_size)
  {
    rank = communicators().pipelineRank();
    nrank = block.pipelineDepth();
  }


  
  FloatType loss(const InputType &x, const OutputType &y){
    int dim = y.size(0);
    int global_batch_size = y.size(1);

    assert( global_batch_size % call_batch_size == 0);
    int navg = global_batch_size / call_batch_size;

    int x_bdims[InputDimension]; memcpy(x_bdims, x.sizeArray(), InputDimension*sizeof(int));
    int y_bdims[OutputDimension]; memcpy(y_bdims, y.sizeArray(), OutputDimension*sizeof(int));
    
    x_bdims[InputDimension-1] = y_bdims[OutputDimension-1] = call_batch_size;
    
    InputType x_dummy(x_bdims,0.0);
    OutputType y_dummy(y_bdims,0.0);
    
    deriv_store = Vector<FloatType>(nparam, 0.);    

    FloatType out = 0.;

    //compute the total required number of calls
    //3 ranks
    //Value:
    //iter    rank->
    //        0     1      2     ret
    //0                   <0|     -
    //1            <0-    <1|     -
    //2      <0-   <1-    <2|     0
    //3      <1-   <2-    <3|     1
    //etc
    
    //value lag: 3

    //Deriv:
    //iter    rank->
    //        0     1      2     ret
    //0       
    //1       
    //2      |0>   
    //3      |1>   -0>
    //4      |2>   -1>    -0>    0
    //5            -2>    -1>    1
    //6                   -2>    2
    //etc

    //deriv lag: 5

    int dcount = 0; //number of derivatives that have been computed. When this is equal to navg we terminate
    int iter =0;
    while(dcount < navg){
      InputType x_iter = iter < navg ? x.sliceLastDimension(iter * call_batch_size, (iter+1)* call_batch_size - 1) : x_dummy;
      OutputType ypred = block.value(x_iter);
      
      int i_vpipe = iter-(value_lag-1);
      int i_dpipe = iter-(deriv_lag-1);

      //start recording loss
      Matrix<FloatType> y_iter(y_dummy);
      if(rank == 0 && i_vpipe >= 0 && i_vpipe < navg){ //only evaluate on first rank where ypred is meaningful
	y_iter = y.sliceLastDimension(i_vpipe * call_batch_size, (i_vpipe+1) * call_batch_size - 1);
	FloatType dloss = cost.loss(y_iter, ypred);
	out += dloss;
      }
	
      //once we start getting values back we need to feed them back to the derivs
      if(i_vpipe >= 0){
	OutputType layer_deriv = rank == 0 ? cost.layer_deriv(y_iter , ypred) : OutputType();
	Vector<FloatType> cost_deriv(nparam,0.);    //zero initialize
	block.deriv(cost_deriv, std::move(layer_deriv)); //inputs ignored apart from on rank 0
	
	if(i_dpipe >= 0){
	  ++dcount;
	  deriv_store += cost_deriv;
	}
      }

      ++iter;
    }

    //Make sure all the ranks get the right output and derivative
    commsBroadcast(deriv_store, 0, communicators().pipelineCommunicator());
    commsBroadcast(&out, 1, 0, communicators().pipelineCommunicator());

    deriv_store *= FloatType(1./navg);
    return out / navg;
  }
  
  Vector<FloatType> deriv() const{ return deriv_store; }

  void update(const Vector<FloatType> &new_params){
    block.update(new_params);
  }
  void step(const Vector<FloatType> &derivs, FloatType eps){
    block.step(derivs,eps);
  }
  int nparams() const{ return nparam; }

  Vector<FloatType> getParams() const{
    Vector<FloatType> out(nparams());
    block.getParams(out);
    return out;
  }

  Matrix<FloatType> predict(const Matrix<FloatType> &x){
    int global_batch_size = x.size(1);

    assert( global_batch_size % call_batch_size == 0);
    int navg = global_batch_size / call_batch_size;

    Matrix<FloatType> x_dummy(x.size(0),call_batch_size,0.0);

    int iters = navg + value_lag -1; //exit when i_vpipe = iter-(value_lag-1) = navg, i.e  iter == navg + (value_lag-1)

    Matrix<FloatType> out;
    
    for(int iter=0;iter<iters;iter++){
      Matrix<FloatType> x_iter = iter < navg ? peekColumns(x,iter * call_batch_size, (iter+1)* call_batch_size - 1) : x_dummy;
      Matrix<FloatType> ypred = block.value(x_iter);

      if(iter == 0 && !rank) out = Matrix<FloatType>(ypred.size(0),global_batch_size);
      
      int i_vpipe = iter-(value_lag-1);      
      if(i_vpipe >= 0 && !rank)
	pokeColumns(out,i_vpipe * call_batch_size, (i_vpipe+1) * call_batch_size - 1, ypred);
    }

    int osize[2] = {out.size(0),out.size(1)};
    commsBroadcast(osize, 2, 0, communicators().pipelineCommunicator());
    if(rank != 0) out = Matrix<FloatType>(osize[0],osize[1]);    
    
    //Make sure all the ranks get the right output
    commsBroadcast(out, 0, communicators().pipelineCommunicator());

    return out;
  }
  //note this call is inefficient as we need to do call_batch_size work for only one data. Use the matrix version if this matters
  Vector<FloatType> predict(const Vector<FloatType> &x){
    Matrix<FloatType> b(x.size(0),call_batch_size,0.);  
    pokeColumn(b,0,x);
    return peekColumn(predict(b),0);    
  }
  
};



//This is a pipelined loss function wrapper. Using it requires careful coordination as the returned value is not the loss for the provided inputs but a delayed result
//from an earlier iteration. Specifically, the output of loss will be the loss for the call valueLag() calls previously, and that for deriv derivLag() calls previously
template<typename PipelineBlockType, typename CostFunc>
class PipelineCostFuncWrapper{
  typedef typename PipelineBlockType::FloatType FloatType;
  typedef typename PipelineBlockType::InputType InputType; //overall input type
  typedef typename PipelineBlockType::OutputType OutputType; //overall output type
  static_assert( std::is_same<OutputType, typename CostFunc::DataType>::value == true );
  
  //types for this rank's block
  typedef typename PipelineBlockType::BlockInputType BlockInputType;
  typedef LAYEROUTPUTTYPE(PipelineBlockType) BlockOutputType; 

  enum { InputDimension = InputType::Dimension, OutputDimension = OutputType::Dimension };
  
  PipelineBlockType &block;

  RingBuffer<OutputType> yval_buf_v;//buffered yvalues for calls to "loss"
  size_t calls;

  OutputType ypred; //predicted value
  OutputType yval; //yval associated with ypred
  
  CostFunc cost;
  int nparam;
  int value_lag;
  int deriv_lag;

  int rank;
public:
  PipelineCostFuncWrapper(PipelineBlockType &block, const CostFunc &cost = CostFunc()): cost(cost), block(block), nparam(block.nparams()),
											value_lag(block.valueLag()), deriv_lag(block.derivLag()),
											yval_buf_v(block.valueLag()),
											calls(0){
    rank = communicators().pipelineRank();
  }


  //On rank 0, return the delayed loss as out.first, with a delay of value_lag calls.
  //If insufficient calls have been made to saturate the pipeline, out.second will be false, otherwise true (on all ranks)
  std::pair<FloatType,bool> loss(const OutputType &x, const OutputType &y){
    ++calls;
    int dim = y.size(0);
    int batch_size = y.size(1);
    
    //<i- (<0-, <1- etc): item i in 'prev_in' at start of iteration, perform action and send left in this iter
    //<i|                : take item i from input 'in', perform action and send left in this iter
    //iter    rank->
    //        0     1      2
    //0                   <0|
    //1            <0-    <1|
    //2      <0-   <1-    <2|
    //3      <1-   <2-    <3|
    //etc
    //value_lag = 3 = nrank
    OutputType ycp(y);
    yval_buf_v.push(std::move(ycp));
   
    ypred = block.value(x);
    
    if(calls < value_lag) return std::pair<FloatType,bool>(-1.,false);
    else{ //yval not initialized until ring buffer is full
      yval = yval_buf_v.pop();
      return std::pair<FloatType,bool>( rank == 0 ? cost.loss(yval,ypred) : 0. , true );
    }
  }

  //On rank 0, return the delayed derivative as out.first, with a delay of deriv_lag calls.
  //If insufficient calls have been made to saturate the pipeline, out.second will be false, otherwise true (on all ranks)
  std::pair<Vector<FloatType>, bool> deriv() const{
    //3 ranks
    //Value:
    //iter    rank->
    //        0     1      2
    //0                   <0|
    //1            <0-    <1|
    //2      <0-   <1-    <2|
    //3      <1-   <2-    <3|
    //etc
    
    //value lag: 3

    //Deriv:
    //iter    rank->
    //        0     1      2
    //0       
    //1       
    //2      |0>   
    //3      |1>   -0>
    //4      |2>   -1>    -0>    
    //etc

    //deriv lag: 5

    //Notice rank 0 consumes y,ypred[i] on the same iteration it receives ypred[i], so we don't need a buffer
    if(calls >= value_lag){ //can't call before value_lag because yval uninitialized
      OutputType layer_deriv = rank == 0 ? cost.layer_deriv(yval, ypred) : OutputType(); //inputs ignored on all ranks bar first
      Vector<FloatType> cost_deriv(nparam,0.);    //zero initialize
      block.deriv(cost_deriv, std::move(layer_deriv));
      if(calls < deriv_lag) return std::pair<Vector<FloatType>, bool>( Vector<FloatType>(nparam,-1.), false ); //indicate that these derivs are invalid
      else return std::pair<Vector<FloatType>, bool>( rank != 0 ? Vector<FloatType>(nparam,-1.) : std::move(cost_deriv), true );
    }else return std::pair<Vector<FloatType>, bool>( Vector<FloatType>(nparam,-1.), false ); //indicate that these derivs are invalid
  }
  
  void update(const Vector<FloatType> &new_params){
    block.update(new_params);
  }
  void step(const Vector<FloatType> &derivs, FloatType eps){
    block.step(derivs,eps);
  }
  int nparams(){ return nparam; }

  Vector<FloatType> getParams() const{
    Vector<FloatType> out(nparams());
    block.getParams(out);
    return out;
  }

  int valueLag() const{ return value_lag; }
  int derivLag() const{ return deriv_lag; }
  
};

#define CWRP PipelineCostFuncWrapper<PipelineBlockType, MSEcostFunc<typename PipelineBlockType::OutputType> >
template<typename PipelineBlockType>
auto pipeline_mse_cost(PipelineBlockType &u)->CWRP{
  return CWRP(u);
}
#undef CWRP


struct LockControlWrapper{
  virtual void lock() = 0;
  virtual void unlock() = 0;
  virtual ~LockControlWrapper(){}
};
template<typename FloatType, int Dim>
struct LockControlWrapperTensor: public LockControlWrapper{
  Tensor<FloatType,Dim> const* v;
  LockControlWrapperTensor(Tensor<FloatType,Dim> const* v): v(v){}
  void lock() override{ v->lock(); }
  void unlock() override{ v->unlock(); }
};  

class PipelineCommunicator{
protected:
  int rank; //current rank
  int next_rank; //rank of next comm layer, -1 indicates this is the last
  int prev_rank; //rank of previous comm layer, -1 indicates this is the first
  int pipeline_depth; //number of ranks in the pipeline
  bool is_first;
  bool is_last;
  
public:
  PipelineCommunicator(){
    //Prepare rank information
    rank = communicators().pipelineRank();
    
    int nranks = communicators().pipelineNrank();

    next_rank = rank+1;
    prev_rank = rank-1;
    if(next_rank == nranks) next_rank = -1;
    pipeline_depth = nranks;

    is_first = prev_rank == -1;
    is_last = next_rank == -1;
  }

  int pipelineDepth() const{ return pipeline_depth; }

  //We want to ensure that managed objects aren't evicted while async MPI operations are happening
  struct CommsRequest{
    std::unique_ptr<LockControlWrapper> v;   
    MPI_Request req;
    
    template<typename FloatType, int Dim>
    CommsRequest(MPI_Request r, const Tensor<FloatType,Dim> &vv): req(r), v(new LockControlWrapperTensor(&vv)){
      v->lock();
    }    
  };
  void waitAll(const std::vector<CommsRequest> &reqs){
    std::vector<MPI_Request> rm(reqs.size());
    for(int i=0;i<reqs.size();i++)
      rm[i] = reqs[i].req;
    assert( MPI_Waitall(rm.size(), rm.data(), MPI_STATUSES_IGNORE) == MPI_SUCCESS );
    for(int i=0;i<reqs.size();i++)
      reqs[i].v->unlock();    
  }
  template<typename T>
  inline static CommsRequest send(const T &mat, int to){
    autoView(mat_v,mat,HostRead);
    MPI_Request req;		
    assert( MPI_Isend(mat_v.data(), mat_v.data_len(), getMPIdataType<typename T::FloatType>(), to, 0, communicators().pipelineCommunicator(), &req) == MPI_SUCCESS );
    return CommsRequest(req,mat);
  }
  template<typename T>
  inline static CommsRequest recv(T &mat, int from){
    MPI_Request req;
    autoView(mat_v,mat,HostWrite);	
    assert( MPI_Irecv(mat_v.data(), mat_v.data_len(), getMPIdataType<typename T::FloatType>(), from, 0, communicators().pipelineCommunicator(), &req) == MPI_SUCCESS );
    return CommsRequest(req,mat);
  }

  template<typename T, typename U>
  void passLeft(std::vector<CommsRequest> &reqs,
		T const* send_bulk, T const *send_last,
		U* recv_first, U* recv_bulk) const{
    if(pipeline_depth == 1) assert(0); //todo: make exception when T==U
    
    if(is_last) reqs.push_back(send(*send_last, prev_rank));
    else if(!is_first) reqs.push_back(send(*send_bulk, prev_rank));

    if(is_first) reqs.push_back(recv(*recv_first, next_rank));
    else if(!is_last) reqs.push_back(recv(*recv_bulk, next_rank));
  }
  template<typename T, typename U>
  void passRight(std::vector<CommsRequest> &reqs,
		T const* send_first, T const *send_bulk,
		U* recv_bulk, U* recv_last) const{
    if(pipeline_depth == 1) assert(0);
    
    if(is_first) reqs.push_back(send(*send_first, next_rank));
    else if(!is_last) reqs.push_back(send(*send_bulk, next_rank));

    if(is_last) reqs.push_back(recv(*recv_last, prev_rank));
    else if(!is_first) reqs.push_back(recv(*recv_bulk, prev_rank));
  }

  template<typename T>
  void passLeftLastToFirst(std::vector<CommsRequest> &reqs,
			   T const* send_last, T *recv_first){
    if(pipeline_depth == 1){
      *recv_first = *send_last; return;
    }
    if(is_last) reqs.push_back(send(*send_last,0));
    else if(is_first) reqs.push_back(recv(*recv_first,pipeline_depth-1));
  }
  template<typename T>
  void passRightFirstToLast(std::vector<CommsRequest> &reqs,
			    T const* send_first, T *recv_last){
    if(pipeline_depth == 1){
      *recv_last = *send_first; return;
    }
    if(is_first) reqs.push_back(send(*send_first,pipeline_depth-1));
    else if(is_last) reqs.push_back(recv(*recv_last,0));
  }
};
  

template<typename BlockStore, typename InputType_, typename OutputType_>
class PipelineBlock: public PipelineCommunicator{
public:
  typedef typename BlockStore::type::FloatType FloatType;
  typedef typename BlockStore::type::InputType BlockInputType;  //not necessarily the data input type, but the tensor type entering as input to this block
  typedef LAYEROUTPUTTYPE(typename BlockStore::type) BlockOutputType;
  
  typedef InputType_ InputType; //overall data input type (=BlockInputType on last rank)
  typedef OutputType_ OutputType; //overall model output type (=BlockOutputType on first rank)
  
  enum { BlockInputDimension = BlockInputType::Dimension, BlockOutputDimension = BlockOutputType::Dimension };
private:  
  BlockStore block; //this chain should terminate on an InputLayer. This represents the work done on this rank
  
  int block_output_dims[BlockOutputDimension]; //features of this block
  int block_input_dims[BlockInputDimension]; //features of output of block to the right
  
  int nparam; //total #params
  int stage_off; //offset within parameter vector associated with this block
  
  BlockInputType prev_block_in; //input for forwards propagation

  //storage for backpropagation
  BlockOutputType prev_above_deriv;
  Vector<FloatType> prev_cost_deriv_passright;  

  int dcalls;

  //circumvent the typing system for ranks where the function will and should never be called
  template<typename OutType, typename B, typename std::enable_if<!std::is_same< typename std::decay<B>::type, OutType>::value, int>::type = 0>
  inline OutType get_as(B &&v){ assert(0); return OutType(); }
  template<typename OutType, typename B, typename std::enable_if<std::is_same< typename std::decay<B>::type, OutType>::value, int>::type = 0>
  inline OutType get_as(B &&v){ return std::move(v); }

  template<typename OutType, typename B, typename std::enable_if<!std::is_same< typename std::decay<B>::type ,OutType>::value, int>::type = 0>
  inline const OutType & get_as(const B &v){ assert(0); static OutType o; return o; }
  template<typename OutType, typename B, typename std::enable_if<std::is_same< typename std::decay<B>::type ,OutType>::value, int>::type = 0>
  inline const OutType & get_as(const B &v){ return v; }
  
public:
  
  PipelineBlock(BlockStore &&_block,
		int const* block_output_dims_,
		int const* block_input_dims_): block(std::move(_block)), dcalls(0){
    memcpy(block_output_dims, block_output_dims_, BlockOutputDimension*sizeof(int));
    memcpy(block_input_dims, block_input_dims_, BlockInputDimension*sizeof(int));    

    //Compute parameter information
    std::vector<int> block_params(pipeline_depth, 0);
    block_params[rank] = block.v.nparams();

    commsReduce(block_params.data(), pipeline_depth, communicators().pipelineCommunicator());

    //Compute block param offset
    stage_off = 0;
    for(int i=0;i<rank;i++) stage_off += block_params[i];

    //Compute total params
    nparam = 0;
    for(int i=0;i<pipeline_depth;i++) nparam += block_params[i];
    
    //Setup storage
    prev_block_in = BlockInputType(block_input_dims, 0.);
    prev_above_deriv = BlockOutputType(block_output_dims, 0.); //dcost/dout_i  from layer above
    prev_cost_deriv_passright = Vector<FloatType>(nparam,0.); //partially-populated cost deriv from layer above
    
    //Tell the block to resize its input buffers accordingly (cf below)
    block.v.resizeInputBuffer(2*rank+1);
  }

  PipelineBlock(const PipelineBlock &r) = delete;
  PipelineBlock(PipelineBlock &&r) = default;
  
  int nparams() const{ return nparam; }

  //Amount of iterations at which you get the return value for the first item back
  //i.e.  iteration  i -> i-(value_lag - 1)    with i=0,1,2...
  int valueLag() const{ return pipeline_depth; }
  //Amount of iterations at which you get the derivative for the first item back
  //i.e.  iteration  i -> i-(deriv_lag - 1)    with i=0,1,2...
  int derivLag() const{ return 2*pipeline_depth - 1; }
 
  //input is ignored for all ranks bar the last
  //we return the output of this block but it is only ultimately used on the first rank
  OutputType value(const InputType &in){
    std::vector<CommsRequest> reqs;

    //<i- (<0-, <1- etc): item i in 'prev_in' at start of iteration, perform action and send left in this iter
    //<i|                : take item i from input 'in', perform action and send left in this iter

    //iter    rank->
    //        0     1      2
    //0                   <0|
    //1            <0-    <1|
    //2      <0-   <1-    <2|
    //3      <1-   <2-    <3|
    //etc

    BlockOutputType out = block.v.value(is_last ? get_as<BlockInputType>(in) : prev_block_in); //last block takes data input in, otherwise we just consume what was previously passed up
    passLeft(reqs,                      &out,        &out,
	          &prev_block_in, &prev_block_in);
    waitAll(reqs);

    if(is_first) return get_as<OutputType>(out); //OutputType == BlockOutputType on first rank
    else return OutputType();
  }
  //inputs are ignored for all ranks bar the first
  void deriv(Vector<FloatType> &cost_deriv, OutputType &&above_deriv){
    ++dcalls;
    std::vector<CommsRequest> reqs;

    //For reverse differentiation we need to wait for a full value pipeline
    //As each layer needs the input value from the layer below to compute its derivative we need to buffer these appropriately

    //2 ranks
    //Value:
    //iter    rank->
    //        0     1
    //0            <0|
    //1      <0-   <1|
    //2      <1-   <2|
    //3      <2-   <3|
    //etc

    //value lag: 2
    
    //Deriv:
    //iter    rank->
    //        0     1    
    //0       
    //1      |0>   
    //2      |1>   -0>
    //3      |2>   -1>   
    //etc

    //deriv lag: 3
    
    //3 ranks
    //Value:
    //iter    rank->
    //        0     1      2
    //0                   <0|
    //1            <0-    <1|
    //2      <0-   <1-    <2|
    //3      <1-   <2-    <3|
    //etc
    
    //value lag: 3

    //Deriv:
    //iter    rank->
    //        0     1      2
    //0       
    //1       
    //2      |0>   
    //3      |1>   -0>
    //4      |2>   -1>    -0>    
    //etc

    //deriv lag: 5


    //4 ranks
    //Value:
    //iter    rank->
    //        0     1      2     3
    //0                         <0|
    //1                  <0-    <1|
    //2            <0-   <1-    <2|
    //3      <0-   <1-   <2-    <3|
    //4      <1-   <2-   <3-    <4|
    //etc
    
    //value lag: 4

    //Deriv:
    //iter    rank->
    //        0     1      2     3
    //0       
    //1
    //2
    //3      |0>   
    //4      |1>   -0>
    //5      |2>   -1>    -0>
    //6      |3>   -2>    -1>    -0>
    //7      |4>   -3>    -2>    -1>
    //etc

    //deriv lag: 7

    //value lag: nrank
    //deriv lag: 2*nrank - 1

    
    
    //To work out the buffering reqs we track what iter the derivs and values are computed for a particular item
    //For item 0:
    //Layer    Val-iter  Deriv-iter   
    // 0          2          2        
    // 1          1          3        
    // 2          0          4

    //For item 1:
    //Layer    Val-iter  Deriv-iter   
    // 0          3          3        
    // 1          2          4        
    // 2          1          5


    //Layer 0
    //Iter     Append    Consume
    // 0         
    // 1         
    // 2         0        0
    // 3         1        1
    // 4         2        2
    //lag = 0
    //buf_sz = 1
    
    //Layer 1
    //Iter     Append    Consume
    // 0         
    // 1         0
    // 2         1
    // 3         2         0
    // 4         3         1
    //lag = 2
    //buf_sz = 3
    
    //Layer 2
    //Iter     Append    Consume
    // 0         0
    // 1         1
    // 2         2
    // 3         3         
    // 4         4         0
    // 5         5         1
    //lag = 4
    //buf_sz = 5

    //buf_sz = 2*layer_idx + 1
    
    
    //compute layer derivative and fill in cost derivative vector
    //if this is the first rank, fill the input cost_deriv, else we append it to the deriv vector received last call
    BlockInputType layer_deriv; //layer deriv to send right //TODO: Handle for non-Matrix InputType
    Vector<FloatType> pass_cost_deriv(is_first ? cost_deriv : prev_cost_deriv_passright); //cost deriv to send right

    block.v.deriv(pass_cost_deriv, stage_off, is_first ? get_as<BlockOutputType>(std::move(above_deriv)) : std::move(prev_above_deriv), &layer_deriv);

    //send layer deriv to right if !last
    prev_above_deriv = BlockOutputType(block_output_dims); //we consumed this matrix above so we need to recreate it!
    passRight(reqs, &layer_deriv, &layer_deriv,
                                 &prev_above_deriv, &prev_above_deriv); 
    
    //send new cost deriv to right if !last
    passRight(reqs, &pass_cost_deriv, &pass_cost_deriv,
	                           &prev_cost_deriv_passright, &prev_cost_deriv_passright);

    //if last our cost derivative is fully populated, so we send it back to rank 0 for output
    passLeftLastToFirst(reqs, &pass_cost_deriv, &cost_deriv);

    waitAll(reqs);
  }

  //Update the parameters. It is assumed that all ranks have the same input 'new_params'
  inline void update(const Vector<FloatType> &new_params){
    block.v.update(stage_off, new_params);
  }

  //Step down the gradient. Assumed that all ranks have the same input 'new_params'
  inline void step(const Vector<FloatType> &derivs, FloatType eps){
    block.v.step(stage_off, derivs, eps);
  }    
  //Get the parameters for the complete model. Each rank will get the same result
  inline void getParams(Vector<FloatType> &into) const{
    into = Vector<FloatType>(nparam, 0.);
    block.v.getParams(into, stage_off);
    commsReduce(into, communicators().pipelineCommunicator());
  }  
  
};

template<typename InputType, typename OutputType, typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto pipeline_block(U &&u, int const* block_output_dims, int const* block_input_dims)->PipelineBlock<DDST(u),InputType,OutputType>{
  return PipelineBlock<DDST(u),InputType,OutputType>(std::forward<U>(u), block_output_dims, block_input_dims);
}

