template<typename T>
inline T bitReverse(T in){
  T out;  
  uint8_t* p = (uint8_t*) &in;
  uint8_t* q = (uint8_t*) &out;
  for(size_t i=0;i<sizeof(T);i++)
    q[sizeof(T)-1-i] = BitReverseTable256(p[i]); 
  return out;  
}

template<typename T, typename std::enable_if<!ISLEAF(T), int>::type>
void BinaryWriter::write(const T&v){
  T tmp = do_flip ? bitReverse(v) : v;
  of.write((char const*)&tmp,sizeof(T));    
}

template<typename T, typename U>
void BinaryWriter::write(const std::pair<T,U> &v){
  write(v.first);
  write(v.second);
}

template<typename T>
void BinaryWriter::write(const std::vector<T> &v){
  uint64_t sz = v.size();
  write(sz);
  for(auto const &e : v)
    write(e);
}

template<typename T, int Dim>
void BinaryWriter::write(const Tensor<T,Dim> &t){
  autoView(t_v,t,HostRead);
  write(Dim);
  for(int i=0;i<Dim;i++)
    write(t.size(i));   
  for(size_t i=0;i<t_v.data_len();i++)
    write(t_v.data()[i]);
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

template<typename T, typename std::enable_if<!ISLEAF(T), int>::type>
inline void BinaryReader::read(T &v){
  T tmp; of.read((char*)&tmp, sizeof(T)); assert(of.good());
  v = do_flip ? bitReverse(tmp) : std::move(tmp);
}

template<typename T, typename U>
void BinaryReader::read(std::pair<T,U> &v){
  read(v.first);
  read(v.second);
}

template<typename T>
void BinaryReader::read(std::vector<T> &v){
  uint64_t sz; read(sz);
  v.resize(sz);
  for(auto &e : v)
    read(e);
}

template<typename T, int Dim>
void BinaryReader::read(Tensor<T,Dim> &t){
  int rDim; read(rDim);
  assert(rDim == Dim);
  for(int i=0;i<Dim;i++){
    int sz; read(sz);    
    assert(sz == t.size(i));
  }
  autoView(t_v,t,HostWrite);      
  for(size_t i=0;i<t_v.data_len();i++)
    read(t_v.data()[i]);
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
