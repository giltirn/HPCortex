#include <Comms.hpp>

int main(int argc, char** argv){
  initialize(argc,argv);

  int world_rank = communicators().worldRank();
  
  if(!world_rank) std::cout << "Default setup:" << std::endl << std::flush;
  communicators().reportSetup();

  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!world_rank) std::cout << "Enabling node-level pipelining" << std::endl << std::flush;
  communicators().enableNodePipelining();
  communicators().reportSetup();


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

}
