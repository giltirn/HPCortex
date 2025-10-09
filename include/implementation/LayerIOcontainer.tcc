template<typename FloatType, int Dim>
std::unique_ptr<LayerIOcontainerInitializer> LayerIOcontainerImpl<Tensor<FloatType,Dim> >::getInitializer() const{
  assert(tens);   
  return std::unique_ptr<LayerIOcontainerInitializer>(new LayerIOcontainerTensorInitializer<Dim>(tens->sizeArray()));
}

template<typename FloatType, int Dim>
void LayerIOcontainerImpl<Tensor<FloatType,Dim> >::initialize(const std::unique_ptr<LayerIOcontainerInitializer> &init) {
  LayerIOcontainerTensorInitializer<Dim> const* ip = dynamic_cast<LayerIOcontainerTensorInitializer<Dim> const*>(init.get());
  tens.reset(new Tensor<FloatType,Dim>(ip->dims));
}

template<typename FloatType, int Dim>
CommsRequest LayerIOcontainerImpl<Tensor<FloatType,Dim> >::send(int to, MPI_Comm &comm) {
  assert(tens);
  tens->lock();
  CommsRequest out;
#ifdef USE_GPU_AWARE_MPI
  autoView(tens_v,(*tens),DeviceRead);
#else
  autoView(tens_v,(*tens),HostRead);
#endif
  assert( MPI_Isend(tens_v.data(), tens_v.data_len(), getMPIdataType<FloatType>(), to, 0, comm, &out.req) == MPI_SUCCESS );
  out.post.reset(new PostCommActionCallbackUnlock(tens.get()));    
  return out;
}

template<typename FloatType, int Dim>
CommsRequest LayerIOcontainerImpl<Tensor<FloatType,Dim> >::recv(int from, MPI_Comm &comm) {
  assert(tens);
  tens->lock();
  CommsRequest out;
#ifdef USE_GPU_AWARE_MPI
  autoView(tens_v,(*tens),DeviceWrite);	
#else
  autoView(tens_v,(*tens),HostWrite);
#endif
  assert( MPI_Irecv(tens_v.data(), tens_v.data_len(), getMPIdataType<FloatType>(), from, 0, comm, &out.req) == MPI_SUCCESS );
  out.post.reset(new PostCommActionCallbackUnlock(tens.get()));
  return out;
}

template<typename FloatType, int Dim>
CommsRequest LayerIOcontainerImpl<Tensor<FloatType,Dim> >::sendInitializer(int to, MPI_Comm &comm) {
  assert(tens);
  CommsRequest out;
  assert( MPI_Isend(tens->sizeArray(), Dim, getMPIdataType<int>(), to, 0, comm, &out.req) == MPI_SUCCESS );
  return out;
}

template<typename FloatType, int Dim>
CommsRequest LayerIOcontainerImpl<Tensor<FloatType,Dim> >::recvInitializer(int from, MPI_Comm &comm) {
  CommsRequest out;
  PostCommActionCallbackTensorInitialize<FloatType,Dim> *callback = new PostCommActionCallbackTensorInitialize<FloatType,Dim>(tens);
  
  assert( MPI_Irecv(callback->size, Dim, getMPIdataType<int>(), from, 0, comm, &out.req) == MPI_SUCCESS );
  out.post.reset(callback);
  return out;
}

template<typename FloatType, int Dim>
LayerIOcontainerBase* LayerIOcontainerImpl<Tensor<FloatType,Dim> >::copy() const {
  if(tens) return new LayerIOcontainerImpl<Tensor<FloatType,Dim> >(*tens);
  else return new LayerIOcontainerImpl<Tensor<FloatType,Dim> >();      
}

template<typename FloatType, int Dim>
Tensor<FloatType,Dim>& LayerIOcontainerImpl<Tensor<FloatType,Dim> >::get(){
  assert(tens);
  return *tens;
}

template<typename FloatType, int Dim>
const Tensor<FloatType,Dim>& LayerIOcontainerImpl<Tensor<FloatType,Dim> >::get() const{
  assert(tens);
  return *tens;
}

template<typename FloatType, int Dim>
Tensor<FloatType,Dim> LayerIOcontainerImpl<Tensor<FloatType,Dim> >::remove(){
  assert(tens);
  Tensor<FloatType,Dim>* p = tens.release();
  Tensor<FloatType,Dim> out(std::move(*p));
  delete p;
  return out;
}

template<typename FloatType, int Dim>
LayerIOcontainerBase * LayerIOcontainerImpl<Tensor<FloatType,Dim> >::getMicroBatch(int ubatch_idx, int ubatch_size) const {
  assert(tens);
  Tensor<FloatType,Dim> slice = tens->sliceLastDimension(ubatch_idx*ubatch_size, (ubatch_idx+1)*ubatch_size - 1);
  return new LayerIOcontainerImpl<Tensor<FloatType,Dim> >(std::move(slice));
}

template<typename FloatType, int Dim>
void LayerIOcontainerImpl<Tensor<FloatType,Dim> >::insertMicroBatch(int ubatch_idx, int ubatch_size, LayerIOcontainerBase *from) {
  assert(tens);
  Tensor<FloatType,Dim> &slice = *(dynamic_cast<LayerIOcontainerImpl<Tensor<FloatType,Dim> >* >(from)->tens);
  tens->insertSliceLastDimension(slice, ubatch_idx*ubatch_size, (ubatch_idx+1)*ubatch_size - 1);
}

template<typename FloatType, int Dim>
void LayerIOcontainerImpl<Tensor<FloatType,Dim> >::insertFirstMicroBatch(int ubatch_size, LayerIOcontainerBase *from, int nubatch) {
  Tensor<FloatType,Dim> &slice = *(dynamic_cast<LayerIOcontainerImpl<Tensor<FloatType,Dim> >* >(from)->tens);
  int size[Dim]; memcpy(size,slice.sizeArray(),Dim*sizeof(int));
  size[Dim-1] = nubatch*ubatch_size;
  tens.reset(new Tensor<FloatType,Dim>(size));
  tens->insertSliceLastDimension(slice, 0, ubatch_size - 1);
}

template<typename FloatType, int Dim>
int LayerIOcontainerImpl<Tensor<FloatType,Dim> >::batchDimSize() const {
  assert(tens);
  return tens->size(Dim-1);
}

template<typename Cond>
void pipelinePassLeftConditional(std::vector<CommsRequest> &reqs,
	       LayerIOcontainer &to, LayerIOcontainer &from, const Cond &send_rank_cond){
  int me = communicators().pipelineRank();
  if(me != 0 && send_rank_cond(me))
    reqs.push_back(from.send(me-1, communicators().pipelineCommunicator()));
  
  if(me != communicators().pipelineNrank()-1 && send_rank_cond(me+1))
    reqs.push_back(to.recv(me+1, communicators().pipelineCommunicator()));
}

template<typename Cond>
void pipelinePassRightConditional(std::vector<CommsRequest> &reqs,
	       LayerIOcontainer &to, LayerIOcontainer &from, const Cond &send_rank_cond){
  int me = communicators().pipelineRank();
  if(me != communicators().pipelineNrank()-1 && send_rank_cond(me))
    reqs.push_back(from.send(me+1, communicators().pipelineCommunicator()));
  
  if(me != 0 && send_rank_cond(me-1))
    reqs.push_back(to.recv(me-1, communicators().pipelineCommunicator()));
}
