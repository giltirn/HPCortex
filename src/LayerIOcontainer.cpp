#include<LayerIOcontainer.hpp>

void pipelineSendRecv(std::vector<CommsRequest> &reqs,
	      LayerIOcontainer &to, LayerIOcontainer &from,
	      int rank_to, int rank_from){
    int me = communicators().pipelineRank();
    if(me == rank_from) reqs.push_back(from.send(rank_to, communicators().pipelineCommunicator()));
    else if(me == rank_to) reqs.push_back(to.recv(rank_from, communicators().pipelineCommunicator()));
}

void pipelinePassRight(std::vector<CommsRequest> &reqs,
	       LayerIOcontainer &to, LayerIOcontainer &from){
  int me = communicators().pipelineRank();
  if(me != communicators().pipelineNrank()-1)
    reqs.push_back(from.send(me+1, communicators().pipelineCommunicator()));
  
  if(me != 0)
    reqs.push_back(to.recv(me-1, communicators().pipelineCommunicator()));
}

void pipelinePassLeft(std::vector<CommsRequest> &reqs,
	       LayerIOcontainer &to, LayerIOcontainer &from){
  int me = communicators().pipelineRank();
  if(me != 0)
    reqs.push_back(from.send(me-1, communicators().pipelineCommunicator()));
  
  if(me != communicators().pipelineNrank()-1)
    reqs.push_back(to.recv(me+1, communicators().pipelineCommunicator()));
}

void pipelineSendRecvInitializer(std::vector<CommsRequest> &reqs,
			 LayerIOcontainer &to, LayerIOcontainer &from,
			 int rank_to, int rank_from){
  int me = communicators().pipelineRank();
  if(me == rank_from) reqs.push_back(from.sendInitializer(rank_to, communicators().pipelineCommunicator()));
  else if(me == rank_to) reqs.push_back(to.recvInitializer(rank_from, communicators().pipelineCommunicator()));
}
