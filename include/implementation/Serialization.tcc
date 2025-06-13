template<typename T>
inline T bitReverse(T in){
  T out;  
  uint8_t* p = (uint8_t*) &in;
  uint8_t* q = (uint8_t*) &out;
  for(size_t i=0;i<sizeof(T);i++)
    q[sizeof(T)-1-i] = BitReverseTable256(p[i]); 
  return out;  
}

template<typename T, int Dim>
void BinaryWriter::write(const Tensor<T,Dim> &t){
  autoView(t_v,t,HostRead);
  writeValue(Dim);
  for(int i=0;i<Dim;i++)
    writeValue(t.size(i));   
  for(size_t i=0;i<t_v.data_len();i++)
    writeValue(t_v.data()[i]);
  assert(of.good());
}

//write a model that is not wrapped by a loss function wrapper
template<typename Model, typename std::enable_if<ISLEAF(Model), int>::type>
void BinaryWriter::write(const Model &model){
  Vector<typename Model::FloatType> p(model.nparams());
  model.getParams(p,0);
  write(p);
}
//write a model wrapped by a loss function wrapper
template<typename Store,typename CostFunc>
void BinaryWriter::write(const CostFuncWrapper<Store,CostFunc> &model){
  write( model.getParams() );    
}


template<typename T, int Dim>
void BinaryReader::read(Tensor<T,Dim> &t){
  assert(readValue<int>() == Dim);
  for(int i=0;i<Dim;i++)
    assert(readValue<int>() == t.size(i));
    
  autoView(t_v,t,HostWrite);      
  for(size_t i=0;i<t_v.data_len();i++)
    t_v.data()[i] = readValue<T>();
}

  
//read a model that is not wrapped by a loss function wrapper
template<typename Model, typename std::enable_if<ISLEAF(Model), int>::type>
void BinaryReader::read(Model &model){
  Vector<typename Model::FloatType> p(model.nparams());
  read(p);
  model.update(0,p);   
}
//read a model wrapped by a loss function wrapper
//write a model wrapped by a loss function wrapper
template<typename Store,typename CostFunc>
void BinaryReader::read(CostFuncWrapper<Store,CostFunc> &model){
  Vector<typename CostFuncWrapper<Store,CostFunc>::FloatType> p(model.nparams());
  read(p);
  model.update(p);   
}
