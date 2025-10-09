template<typename LayerOutputType, typename BelowStore>
void PipelineBlockLayer<LayerOutputType,BelowStore>::initialize(){
  if(initialized) return;
  assert(rank_block);
  rank_block_nparam[rank] = rank_block->nparams();
  commsReduce(rank_block_nparam.data(),rank_block_nparam.size(), communicators().pipelineCommunicator());

  for(int r=0;r<pipeline_depth;r++){
    if(r==rank) rank_param_offset = nparam;

    if(!rank) std::cout << "Rank params " << r << " : " << rank_block_nparam[r] << std::endl;
    nparam += rank_block_nparam[r];
  }
  if(!rank) std::cout << "Total params : " << nparam << std::endl;

  //note: the ordering of parameters is *top-down*, so the rank's parameter offset counts down
  rank_param_offset = nparam;
  for(int r=0;r<=rank;r++)
    rank_param_offset -= rank_block_nparam[r];

  initialized = true;
}


template<typename LayerOutputType, typename BelowStore>
void PipelineBlockLayer<LayerOutputType,BelowStore>::gatherParameterVector(int to_off, Vector<FloatType> &vec_to, Vector<FloatType> &vec_from) const{
  if(rank != 0){
#ifdef USE_GPU_AWARE_MPI
    autoView(vec_from_v, vec_from, DeviceRead);
#else
    autoView(vec_from_v, vec_from, HostRead);
#endif
    MPI_Request req;
    assert( MPI_Isend(vec_from_v.data(), vec_from_v.data_len(), getMPIdataType<FloatType>(),
		      0, 0, communicators().pipelineCommunicator(), &req) == MPI_SUCCESS );
    assert(MPI_Wait(&req, MPI_STATUS_IGNORE) == MPI_SUCCESS);
  }else{
    //note: the ordering of parameters is *top-down*, so for the pipeline we need to put the last rank first in the output
#ifdef USE_GPU_AWARE_MPI
    autoView(vec_to_v, vec_to, DeviceReadWrite);
#else
    autoView(vec_to_v, vec_to, HostReadWrite);
#endif
    std::vector<MPI_Request> mr(pipeline_depth-1);
    FloatType* cdp = vec_to_v.data() + to_off;
    for(int r=1;r<pipeline_depth;r++){
      int from_rank = pipeline_depth-r;
      assert( MPI_Irecv(cdp , rank_block_nparam[from_rank],
			getMPIdataType<FloatType>(), from_rank, 0, communicators().pipelineCommunicator(), &mr[r-1]) == MPI_SUCCESS );
      cdp += rank_block_nparam[from_rank];
    }
    assert(MPI_Waitall(pipeline_depth-1, mr.data(), MPI_STATUSES_IGNORE) == MPI_SUCCESS);

#ifdef USE_GPU_AWARE_MPI
    autoView(vec_from_v, vec_from, DeviceRead);
    acceleratorCopyDeviceToDevice(cdp, vec_from_v.data(), vec_from_v.data_len()*sizeof(FloatType));
#else
    autoView(vec_from_v, vec_from, HostRead);
    memcpy(cdp, vec_from_v.data(), vec_from_v.data_len()*sizeof(FloatType));
#endif

  }
}

template<typename LayerOutputType, typename BelowStore>
LayerOutputType PipelineBlockLayer<LayerOutputType,BelowStore>::value(const InputType &x, EnableDeriv enable_deriv){
  initialize();
   
  LayerIOcontainer below_in(below.v.value(x,enable_deriv)); //everyone call below.value but it need only be valid content on rank 0
  int input_batch_dim_size = below_in.batchDimSize();
    
  assert(input_batch_dim_size % ubatch_size == 0);
    
  int nubatch = input_batch_dim_size/ubatch_size;    
  rank_block->resizeInputBuffer(nubatch);
    
  assert(nubatch >= pipeline_depth);

  std::vector<CommsRequest> reqs;

  LayerIOcontainer rank_block_in, rank_block_out, pipeline_out;
  if(!is_first) rank_block->setInputType(rank_block_in);
  pipeline_out.setType<LayerOutputType>();
    
  int ubatch_idx_in=0;
  //"prime" the pipeline
  for(int prime=0; prime < pipeline_depth; prime++){
    if(is_first)
      rank_block_in = below_in.getMicroBatch(ubatch_idx_in, ubatch_size);
    ++ubatch_idx_in;
      
    //Which rank has data depends on the iteration
    //eg for 4 ranks
    //0  :   0 <--from below
    //1  :   0,1
    //2  :   0,1,2
    //3  :   0,1,2,3
    if(rank <= prime) rank_block_out = rank_block->blockValue(rank_block_in,enable_deriv);

    //Ranks that were just called for the first time need to send their output initializers to the right
    if(prime != pipeline_depth-1){
      if(!rank_block_in_init){	
	pipelineSendRecvInitializer(reqs, rank_block_in, rank_block_out, prime+1, prime);
	waitAll(reqs);
      }else if(rank == prime+1) rank_block_in.initialize(rank_block_in_init);	  
    }

    //All active ranks need to send their output to the right
    pipelinePassRightConditional(reqs, rank_block_in, rank_block_out, [=](int send_rank){ return send_rank <= prime; });
    waitAll(reqs);
  }
  //after priming, rank_block_out on the last rank should be the first pipeline output
  if(is_last) pipeline_out.insertFirstMicroBatch(ubatch_size, rank_block_out, nubatch);    

  //record initializer for next call
  if(!rank_block_in_init) rank_block_in_init = rank_block_in.getInitializer();
    
  //steady state
  int ubatch_idx_out = 1;
  for(int iter = pipeline_depth; iter < nubatch; iter++){
    if(is_first)
      rank_block_in = below_in.getMicroBatch(ubatch_idx_in, ubatch_size);
    ++ubatch_idx_in;
      
    rank_block_out = rank_block->blockValue(rank_block_in, enable_deriv);
    pipelinePassRight(reqs, rank_block_in, rank_block_out);
    waitAll(reqs);
    if(is_last) pipeline_out.insertMicroBatch(ubatch_idx_out, ubatch_size, rank_block_out);
    ++ubatch_idx_out;
  }

  //at this point we should have no further input batches
  assert(ubatch_idx_in == nubatch);
    
  //"drain" the pipeline
  for(int drain=0;drain < pipeline_depth-1; drain++){
    if(rank > drain) rank_block_out = rank_block->blockValue(rank_block_in, enable_deriv);
    pipelinePassRightConditional(reqs, rank_block_in, rank_block_out, [=](int send_rank){ return send_rank > drain; }  );
    waitAll(reqs);
    if(is_last) pipeline_out.insertMicroBatch(ubatch_idx_out, ubatch_size, rank_block_out);
    ++ubatch_idx_out;
  }
  //should now be done
  assert(ubatch_idx_out == nubatch);

  value_flops = rank_block->blockFLOPS(0) * nubatch;
    
  //Communicate result back to pipeline leader and return
  if(!pipeline_out_init){
    for(int r=0;r<pipeline_depth-1;r++)
      pipelineSendRecvInitializer(reqs, pipeline_out, pipeline_out, r, pipeline_depth-1); //all ranks have output of correct *size*
    waitAll(reqs);
    pipeline_out_init = pipeline_out.getInitializer();
  }else if(!is_last) pipeline_out.initialize(pipeline_out_init);
    
  pipelineSendRecv(reqs, pipeline_out, pipeline_out, 0, pipeline_depth-1); //only rank 0 has actual output
  waitAll(reqs);
  return pipeline_out.remove<LayerOutputType>();
}

template<typename LayerOutputType, typename BelowStore>
int PipelineBlockLayer<LayerOutputType,BelowStore>::deriv(Vector<FloatType> &cost_deriv, int off, LayerOutputType &&_above_deriv, InputType* input_above_deriv_return) const{    
  LayerIOcontainer pipeline_above_deriv(std::move(_above_deriv));  //all ranks store the rvalue but only the pipeline leader is required to have valid input
  Vector<FloatType> rank_block_cost_derivs(rank_block->nparams(), 0.); //accumulate
   
  int batch_dim_size = pipeline_above_deriv.batchDimSize();
  assert(batch_dim_size % ubatch_size == 0);
    
  int nubatch = batch_dim_size/ubatch_size;
    
  //communicate above deriv to top of pipeline
  std::vector<CommsRequest> reqs;      
  pipelineSendRecv(reqs, pipeline_above_deriv, pipeline_above_deriv, pipeline_depth-1, 0); //already initialized to correct size by move
  waitAll(reqs);

  LayerIOcontainer rank_above_deriv, rank_input_deriv, pipeline_input_deriv;
  pipeline_input_deriv.setType<LayerInputType>();
  rank_block->setOutputType(rank_above_deriv);
    
  //prime the backwards pass
  int ubatch_idx_in = 0;
  for(int prime=0; prime < pipeline_depth; prime++){
    if(is_last) rank_above_deriv = pipeline_above_deriv.getMicroBatch(ubatch_idx_in, ubatch_size);
    ++ubatch_idx_in;
    if(rank >= pipeline_depth - prime -1){
      Vector<FloatType> cd(rank_block->nparams(),0.);
      rank_block->blockDeriv(cd, rank_above_deriv, rank_input_deriv);
      rank_block_cost_derivs += cd;
    }
    //Ranks that were just called for the first time need to send their output initializers to the left
    if(prime != pipeline_depth-1){
      if(!rank_above_deriv_init){
	pipelineSendRecvInitializer(reqs, rank_above_deriv, rank_input_deriv, pipeline_depth-prime-2, pipeline_depth-prime-1);
	waitAll(reqs);
      }else if(rank == pipeline_depth-prime-2)
	rank_above_deriv.initialize(rank_above_deriv_init);	  
    }
    //All active ranks need to send their output to the left
    pipelinePassLeftConditional(reqs, rank_above_deriv, rank_input_deriv, [=](int send_rank){ return send_rank >= pipeline_depth - prime -1; });
    waitAll(reqs);
  }

  //At this point the first batch of input derivs should have reached rank 0
  if(is_first)
    pipeline_input_deriv.insertFirstMicroBatch(ubatch_size, rank_input_deriv, nubatch);      

  if(!rank_above_deriv_init) rank_above_deriv_init = rank_above_deriv.getInitializer();
    
  //steady state
  int ubatch_idx_out = 1;
  for(int iter = pipeline_depth; iter < nubatch; iter++){
    if(is_last) rank_above_deriv = pipeline_above_deriv.getMicroBatch(ubatch_idx_in, ubatch_size);
    ++ubatch_idx_in;
    Vector<FloatType> cd(rank_block->nparams(),0.);
    rank_block->blockDeriv(cd, rank_above_deriv, rank_input_deriv);
    rank_block_cost_derivs += cd;
      
    pipelinePassLeft(reqs, rank_above_deriv, rank_input_deriv);
    waitAll(reqs);
    if(is_first) pipeline_input_deriv.insertMicroBatch(ubatch_idx_out, ubatch_size, rank_input_deriv);
    ++ubatch_idx_out;
  }

  //at this point we should have no further input batches
  assert(ubatch_idx_in == nubatch);

  //"drain" the pipeline
  for(int drain=0;drain < pipeline_depth-1; drain++){
    if(rank < pipeline_depth-1-drain){
      Vector<FloatType> cd(rank_block->nparams(),0.);
      rank_block->blockDeriv(cd, rank_above_deriv, rank_input_deriv);
      rank_block_cost_derivs += cd;
    }
    pipelinePassLeftConditional(reqs, rank_above_deriv, rank_input_deriv, [=](int send_rank){ return send_rank < pipeline_depth-1-drain; });
    waitAll(reqs);
      
    if(is_first) pipeline_input_deriv.insertMicroBatch(ubatch_idx_out, ubatch_size, rank_input_deriv);
    ++ubatch_idx_out;
  }
  //should now be done
  assert(ubatch_idx_out == nubatch);

  deriv_flops = rank_block->blockFLOPS(1) * nubatch;
    
  //gather the cost derivs to rank 0
  gatherParameterVector(off, cost_deriv, rank_block_cost_derivs);

  //ensure the input deriv has the right size for all ranks
  if(!pipeline_input_deriv_init){
    for(int r=1;r<pipeline_depth;r++)
      pipelineSendRecvInitializer(reqs, pipeline_input_deriv, pipeline_input_deriv, r, 0);
    waitAll(reqs);
    pipeline_input_deriv_init = pipeline_input_deriv.getInitializer();
  }else if(rank !=0) pipeline_input_deriv.initialize(pipeline_input_deriv_init);   

  return below.v.deriv(cost_deriv, off + nparam, pipeline_input_deriv.remove<LayerInputType>(), input_above_deriv_return);
}

template<typename LayerOutputType, typename BelowStore>
inline void PipelineBlockLayer<LayerOutputType,BelowStore>::resizeInputBuffer(size_t to){
  if(to != 1) throw std::runtime_error("Using a pipeline layer inside other pipelines is not currently supported");
  below.v.resizeInputBuffer(to);
}

template<typename LayerOutputType, typename BelowStore>
int PipelineBlockLayer<LayerOutputType,BelowStore>::update(int off, const Vector<FloatType> &new_params){
  Vector<FloatType> np(new_params);
  commsBroadcast(np, 0, communicators().pipelineCommunicator());
  commsBroadcast(&off, 1, 0, communicators().pipelineCommunicator());
  rank_block->blockUpdate(off + rank_param_offset, np);
  return below.v.update(off + nparam, new_params);
}

template<typename LayerOutputType, typename BelowStore>
int PipelineBlockLayer<LayerOutputType,BelowStore>::step(int off, const Vector<FloatType> &derivs, FloatType eps){
  Vector<FloatType> np(derivs);
  commsBroadcast(np, 0, communicators().pipelineCommunicator());
  commsBroadcast(&off, 1, 0, communicators().pipelineCommunicator());
  rank_block->blockStep(off + rank_param_offset, np, eps);
  return below.v.step(off + nparam, derivs, eps);
}

template<typename LayerOutputType, typename BelowStore>
int PipelineBlockLayer<LayerOutputType,BelowStore>::getParams(Vector<FloatType> &into, int off) const{
  Vector<FloatType> rank_params(rank_block->nparams());
  rank_block->blockGetParams(rank_params);
  commsBroadcast(&off, 1, 0, communicators().pipelineCommunicator());
  gatherParameterVector(off, into, rank_params);
  return below.v.getParams(into, off + nparam);
}

template<typename LayerOutputType, typename BelowStore>
size_t PipelineBlockLayer<LayerOutputType,BelowStore>::FLOPS(int value_or_deriv) const{
  uint64_t fl = value_or_deriv == 0 ? value_flops : deriv_flops;
  commsReduce(&fl,1, communicators().pipelineCommunicator());
  fl += below.v.FLOPS(value_or_deriv);
  return fl;
}
