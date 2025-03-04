template<>
inline MPI_Datatype getMPIdataType<double>(){ return MPI_DOUBLE; }

template<>
inline MPI_Datatype getMPIdataType<float>(){ return MPI_FLOAT; }

template<>
inline MPI_Datatype getMPIdataType<int>(){ return MPI_INT; }

template<typename FloatType>
inline void commsReduce(FloatType *data, size_t data_len, const MPI_Comm &comm){
  assert( MPI_Allreduce(MPI_IN_PLACE, data, data_len, getMPIdataType<FloatType>(), MPI_SUM, comm) == MPI_SUCCESS );
}

template<typename FloatType>
inline void commsBroadcast(FloatType* data, size_t data_len, int from_rank, const MPI_Comm &comm){
  assert(MPI_Bcast(data, data_len, getMPIdataType<FloatType>(), from_rank, comm) == MPI_SUCCESS );
}
