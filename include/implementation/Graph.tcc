template<typename FloatType>
AttributedGraphElement<FloatType> & AttributedGraphElement<FloatType>::operator+=(const AttributedGraphElement<FloatType> &r){    
  int nattr = attributes.size();
  assert(r.attributes.size() == nattr);
  for(int a=0;a<nattr;a++)
    attributes[a] += r.attributes[a];
  return *this;
}

template<typename FloatType>
void AttributedGraphElement<FloatType>::insertBatch(const AttributedGraphElement<FloatType> &from, int bidx){
  assert(from.attributes.size() == attributes.size());
  for(int a=0;a<attributes.size();a++)
    batchInsertAttrib(attributes[a], from.attributes[a], bidx);
}

template<typename FloatType>
void AttributedGraphElement<FloatType>::initialize(const std::vector<int> &attr_sizes, int batch_size){
  attributes.resize(attr_sizes.size());
  for(int a=0;a<attr_sizes.size();a++)
    attributes[a] = Matrix<FloatType>(attr_sizes[a], batch_size, FloatType(0.));
}

template<typename FloatType>
std::vector<int> AttributedGraphElement<FloatType>::getAttributeSizes() const{
  std::vector<int> out(attributes.size());
  for(int a=0;a<attributes.size();a++)
    out[a] = attributes[a].size(0);
  return out;
}

template<typename FloatType>
void AttributedGraphElement<FloatType>::setZero(){
  for(auto &attr : attributes){
    autoView(attr_v, attr, DeviceWrite);
    acceleratorMemSet(attr_v.data(),0,attr_v.data_len() * sizeof(FloatType));
  }
}

template<typename FloatType>
int AttributedGraphElement<FloatType>::totalAttribSize() const{
  int out = 0;
  for(auto const &a : attributes) out += a.size(0);
  return out;
}
  

template<typename FloatType>
Edge<FloatType> & Edge<FloatType>::operator+=(const Edge<FloatType> &r){
  assert(r.send_node == send_node && r.recv_node == recv_node);
  this->AttributedGraphElement<FloatType>::operator+=(r);
  return *this;
}

template<typename FloatType>
void Edge<FloatType>::insertBatch(const Edge<FloatType> &from, int bidx){
  assert(from.send_node == send_node && from.recv_node == recv_node);
  this->AttributedGraphElement<FloatType>::insertBatch(from,bidx);
}

template<typename FloatType>
void Edge<FloatType>::initialize(int send_node, int recv_node, const std::vector<int> &attr_sizes, int batch_size){
  this->send_node = send_node;
  this->recv_node = recv_node;
  this->AttributedGraphElement<FloatType>::initialize(attr_sizes,batch_size);
}


template<typename FloatType>
void batchInsertAttrib(Matrix<FloatType> &into, const Matrix<FloatType> &from, int bidx){
  assert(from.size(0) == into.size(0) && from.size(1) == 1);
  autoView(into_v,into,DeviceReadWrite);
  autoView(from_v,from,DeviceRead);
  accelerator_for_gen(0,1,normal(),i,into.size(0),{
      into_v(i,bidx) = from_v(i,0);
    });
}

template<typename FloatType>
Graph<FloatType>::Graph(const GraphInitialize &init): nodes(init.nnode), edges(init.edge_map.size()){
  for(int n=0;n<nodes.size();n++) nodes[n].initialize(init.node_attr_sizes, init.batch_size);
  for(int e=0;e<edges.size();e++) edges[e].initialize(init.edge_map[e].first, init.edge_map[e].second, init.edge_attr_sizes, init.batch_size);
  global.initialize(init.global_attr_sizes, init.batch_size);
}

template<typename FloatType>
GraphInitialize Graph<FloatType>::getInitializer() const{
  GraphInitialize out;
  out.nnode = nodes.size();

  out.node_attr_sizes = nodes[0].getAttributeSizes();
  out.edge_attr_sizes = edges[0].getAttributeSizes();
  
  out.edge_map.resize(edges.size());
  for(int e=0;e<edges.size();e++){
    out.edge_map[e].first = edges[e].send_node;
    out.edge_map[e].second = edges[e].recv_node;
  }

  out.global_attr_sizes = global.getAttributeSizes();
  out.batch_size = global.attributes[0].size(1);
  return out;
}

template<typename FloatType>
template<typename Action>
void Graph<FloatType>::applyToAllAttributes(const Action &act){
  for(auto &n : nodes)
    for(auto &a : n.attributes)
      act(a);
  for(auto &e : edges)
    for(auto &a : e.attributes)
      act(a);
  for(auto &a : global.attributes)
    act(a);
}

template<typename FloatType>
Graph<FloatType> & Graph<FloatType>::operator+=(const Graph<FloatType> &r){
  assert(nodes.size() == r.nodes.size());
  for(int n=0;n<nodes.size();n++)
    nodes[n] += r.nodes[n];
  assert(edges.size() == r.edges.size());
  for(int e=0;e<edges.size();e++)
    edges[e] += r.edges[e];
  global += r.global;
  return *this;
}

template<typename FloatType>
void Graph<FloatType>::insertBatch(const Graph<FloatType> &from, int bidx){
  assert(from.nodes.size() == nodes.size() && from.edges.size() == edges.size());
  for(int n=0;n<nodes.size();n++)
    nodes[n].insertBatch(from.nodes[n],bidx);
  for(int e=0;e<edges.size();e++)
    edges[e].insertBatch(from.edges[e],bidx);
  global.insertBatch(from.global, bidx);
}

template<typename FloatType>
FloatType* stackAttr(FloatType *to_device, const Matrix<FloatType> &attr){
  int batch_size = attr.size(1);
  int attr_size = attr.size(0);
  autoView(attr_v, attr, DeviceRead);
  accelerator_for_2d_gen(1,1, normal(), b, batch_size, i, attr_size,
			 {
			   FloatType* to = to_device + i*batch_size + b;
			   *to = attr_v(i,b);
			 });
  return to_device + attr_size * batch_size;
}
template<typename FloatType>
FloatType const* unstackAttrAdd(Matrix<FloatType> &attr, FloatType const* from_device){
  int batch_size = attr.size(1);
  int attr_size = attr.size(0);
  autoView(attr_v, attr, DeviceReadWrite);
  accelerator_for_2d_gen(1,1, normal(), b, batch_size, i, attr_size,
			 {
			   FloatType const* from = from_device + i*batch_size + b;
			   attr_v(i,b) += *from;
			 });
  return from_device + attr_size * batch_size;
}
template<typename FloatType>
FloatType const* unstackAttr(Matrix<FloatType> &attr, FloatType const* from_device){
  int batch_size = attr.size(1);
  int attr_size = attr.size(0);
  autoView(attr_v, attr, DeviceWrite);
  accelerator_for_2d_gen(1,1, normal(), b, batch_size, i, attr_size,
			 {
			   FloatType const* from = from_device + i*batch_size + b;
			   attr_v(i,b) = *from;
			 });
  return from_device + attr_size * batch_size;
}


template<typename FloatType>
FloatType* stackAttr(FloatType *to_device, const AttributedGraphElement<FloatType> &elem){
  for(auto const &a : elem.attributes) to_device = stackAttr(to_device, a);
  return to_device;
}
template<typename FloatType>
FloatType const* unstackAttrAdd(AttributedGraphElement<FloatType> &elem, FloatType const* from_device){
  for(auto &a : elem.attributes) from_device = unstackAttrAdd(a, from_device);
  return from_device;
}
template<typename FloatType>
FloatType const* unstackAttr(AttributedGraphElement<FloatType> &elem, FloatType const* from_device){
  for(auto &a : elem.attributes) from_device = unstackAttr(a, from_device);
  return from_device;
}


template<typename FloatType>
int flatSize(const Matrix<FloatType> &m){
  return m.size(0)*m.size(1);
}

template<typename FloatType>
int flatSize(const AttributedGraphElement<FloatType> &attr){
  int out = 0;
  for(auto const &a : attr.attributes)
    out += flatSize(a);
  return out;
}

template<typename FloatType>
int flatSize(const Graph<FloatType> &g){
  int out=0;
  for(int n=0;n<g.nodes.size();n++)
    out += flatSize(g.nodes[n]);
  for(int e=0;e<g.edges.size();e++)
    out += flatSize(g.edges[e]);
  out += flatSize(g.global);
  return out;
}

template<typename FloatType>
FloatType* flatten(FloatType *to_host_ptr, const AttributedGraphElement<FloatType> &elem){
  for(auto const &attr : elem.attributes)
    to_host_ptr = flatten(to_host_ptr, attr);
  return to_host_ptr;
}
template<typename FloatType>
FloatType const* unflatten(AttributedGraphElement<FloatType> &elem, FloatType const *in_host_ptr){
  for(auto &attr : elem.attributes)
    in_host_ptr = unflatten(attr, in_host_ptr);
  return in_host_ptr;
}



template<typename FloatType>
FloatType* flatten(FloatType *to_host_ptr, const Graph<FloatType> &graph){
  for(auto const &n : graph.nodes) to_host_ptr = flatten(to_host_ptr, n);
  for(auto const &e : graph.edges) to_host_ptr = flatten(to_host_ptr, e);
  return flatten(to_host_ptr, graph.global);
}
template<typename FloatType>
Vector<FloatType> flatten(const Graph<FloatType> &graph){
  int out_size = flatSize(graph);
  Vector<FloatType> out(out_size);
  autoView(out_v, out, HostWrite);
  flatten(out_v.data(), graph);
  return out;
}

template<typename FloatType>
FloatType const* unflatten(Graph<FloatType> &graph, FloatType const *in_host_ptr){
  for(auto &n : graph.nodes) in_host_ptr = unflatten(n, in_host_ptr);
  for(auto &e : graph.edges) in_host_ptr = unflatten(e, in_host_ptr);
  return unflatten(graph.global, in_host_ptr);
}
template<typename FloatType>
void unflatten(Graph<FloatType> &graph, const Vector<FloatType> &in){
  int g_size = flatSize(graph);
  assert(in.size(0)==g_size);

  autoView(in_v, in, HostRead);
  unflatten(graph, in_v.data());
}
