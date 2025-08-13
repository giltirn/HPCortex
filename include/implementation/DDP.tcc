template<typename FloatType>
void ddpAverage(FloatType* data, size_t len, bool pipeline_bcast){
  //Communicate only on the pipeline leaders
  if(communicators().isPipelineLeader()){
    int nrank = communicators().ddpNrank();
    commsReduce(data,len,communicators().ddpCommunicator());
    for(size_t i=0;i<len;i++) data[i] /= nrank;
  }
  //Broadcast to pipeline members (e.g. for parameter update)
  if(pipeline_bcast && communicators().pipelineNrank()>1){
    commsBroadcast(data, len, 0, communicators().pipelineCommunicator());
  }
}

template<typename FloatType>
void ddpAverage(Vector<FloatType> &v, bool pipeline_bcast){
  if(communicators().ddpNrank() == 1 &&
     (!pipeline_bcast || (pipeline_bcast && communicators().pipelineNrank() == 1) )
     ) return; //skip need to pull data to host if on device
  
  autoView(v_v,v,HostReadWrite);
  ddpAverage(v_v.data(),v_v.data_len(), pipeline_bcast);
}
