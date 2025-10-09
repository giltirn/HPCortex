#include <HPCortex.hpp>

int main(int argc, char** argv){
  initialize(argc,argv);

  int world_rank = communicators().worldRank();
  
  if(!world_rank) std::cout << "Default setup:" << std::endl << std::flush;
  communicators().reportSetup();

  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!world_rank) std::cout << "Enabling node-level pipelining" << std::endl << std::flush;
  communicators().enableNodePipelining();
  communicators().reportSetup();

#ifdef USE_GPU_AWARE_MPI
  {
    std::cout << "Testing GPU-aware MPI between on-node devices" << std::endl;
    int ranks = communicators().pipelineNrank();
    int rank = communicators().pipelineRank();
    int* send = (int*)acceleratorAllocDevice(sizeof(int));
    int* recv = (int*)acceleratorAllocDevice(sizeof(int));
    int val = rank;
    acceleratorCopyToDevice(send, &val, sizeof(int));
    for(int i=0;i<=ranks;i++){
      assert(MPI_Sendrecv(send, 1, MPI_INT, (rank+1) % ranks, 0,
			  recv, 1, MPI_INT, (rank-1+ranks) % ranks, 0,
			  communicators().pipelineCommunicator(), MPI_STATUS_IGNORE) == MPI_SUCCESS );
      acceleratorCopyFromDevice(&val, recv, sizeof(int));
      int expect = (rank -i-1 + 2*ranks) % ranks;
      if(rank==0) std::cout << "Received " << val << " expect " << expect << std::endl;
      assert(val == expect );
      acceleratorCopyDeviceToDevice(send,recv,sizeof(int));
    }
    acceleratorFreeDevice(send);
    acceleratorFreeDevice(recv);
  }
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!world_rank) std::cout << "Dividing ranks into blocks of 2" << std::endl << std::flush;
  communicators().enableColorPipelining(world_rank / 2);
  communicators().reportSetup();


  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!world_rank) std::cout << "All ranks in one pipeline" << std::endl << std::flush;
  communicators().enableGlobalPipelining();
  communicators().reportSetup();

  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!world_rank) std::cout << "Disable all parallelism" << std::endl << std::flush;
  communicators().disableParallelism();
  communicators().reportSetup();


  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!world_rank) std::cout << "Restore default setup" << std::endl << std::flush;
  communicators().enableDDPnoPipelining();
  communicators().reportSetup();


  
  std::cout << "testComms passed"<< std::endl; //TODO: Need actual checks here!!
}
