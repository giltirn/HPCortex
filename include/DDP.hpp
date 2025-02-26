#pragma once
#include <Comms.hpp>
#include <Tensors.hpp>

void ddpAverage(double* data, size_t len, bool pipeline_bcast = false){
  //Communicate only on the pipeline leaders
  if(communicators().isPipelineLeader()){
    int nrank = communicators().ddpNrank();
    assert( MPI_Allreduce(MPI_IN_PLACE, data, len, MPI_DOUBLE, MPI_SUM, communicators().ddpCommunicator()) == MPI_SUCCESS );
    for(size_t i=0;i<len;i++) data[i] /= nrank;
  }
  //Broadcast to pipeline members (e.g. for parameter update)
  if(pipeline_bcast & communicators().pipelineNrank()>1){
    assert(MPI_Bcast(data, len, MPI_DOUBLE, 0, communicators().pipelineCommunicator()) == MPI_SUCCESS );
  }
}
