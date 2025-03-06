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

template<typename FloatType>
inline void commsBroadcast(Vector<FloatType> &v, int from_rank, const MPI_Comm &comm){
  autoView(v_v,v,HostReadWrite);
  commsBroadcast(v_v.data(),v_v.data_len(),from_rank,comm);
}

template<typename FloatType>
inline void commsBroadcast(Matrix<FloatType> &v, int from_rank, const MPI_Comm &comm){
  autoView(v_v,v,HostReadWrite);
  commsBroadcast(v_v.data(),v_v.data_len(),from_rank,comm);
}


template<typename FloatType>
inline void commsReduce(Vector<FloatType> &v, const MPI_Comm &comm){
  autoView(v_v,v,HostReadWrite);
  commsReduce(v_v.data(),v_v.data_len(),comm);
}

