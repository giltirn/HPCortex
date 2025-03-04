template<typename FloatType>
void ddpAverage(FloatType* data, size_t len, bool pipeline_bcast){
  //Communicate only on the pipeline leaders
  if(communicators().isPipelineLeader()){
    int nrank = communicators().ddpNrank();
    commsReduce(data,len,communicators().ddpCommunicator());
    for(size_t i=0;i<len;i++) data[i] /= nrank;
  }
  //Broadcast to pipeline members (e.g. for parameter update)
  if(pipeline_bcast & communicators().pipelineNrank()>1){
    commsBroadcast(data, len, 0, communicators().pipelineCommunicator());
  }
}
