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
Graph<FloatType>::Graph(const GraphInitialize &init): nodes(init.nnode), edges(init.edge_map.size()), global(init.global_attr_size, init.batch_size, FloatType(0.)){
  for(int n=0;n<nodes.size();n++) nodes[n].initialize(init.node_attr_sizes, init.batch_size);
  for(int e=0;e<edges.size();e++) edges[e].initialize(init.edge_map[e].first, init.edge_map[e].second, init.edge_attr_sizes, init.batch_size);
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

  out.global_attr_size = global.size(0);
  out.batch_size = global.size(1);
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
  act(global);      
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
  batchInsertAttrib(global, from.global, bidx);
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
int flatSize(const Matrix<FloatType> &m){
  return m.size(0)*m.size(1);
}

template<typename FloatType>
int flatSize(const std::vector< Matrix<FloatType> > &attr){
  int out = 0;
  for(int a=0;a<attr.size();a++)
    out += flatSize(attr[a]);
  return out;
}

template<typename FloatType>
int flatSize(const Node<FloatType> &n){
  return flatSize(n.attributes);
}

template<typename FloatType>
int flatSize(const Edge<FloatType> &e){
  return flatSize(e.attributes);
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
FloatType* flatten(FloatType *to_host_ptr, const Graph<FloatType> &graph){
  for(int n=0;n<graph.nodes.size();n++)
    for(int a=0;a<graph.nodes[n].attributes.size();a++)
      to_host_ptr = flatten(to_host_ptr, graph.nodes[n].attributes[a]);
  
  for(int e=0;e<graph.edges.size();e++)
    for(int a=0;a<graph.edges[e].attributes.size();a++)
      to_host_ptr = flatten(to_host_ptr, graph.edges[e].attributes[a]);  

  to_host_ptr = flatten(to_host_ptr, graph.global);
  return to_host_ptr;
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
  for(int n=0;n<graph.nodes.size();n++)
    for(int a=0;a<graph.nodes[n].attributes.size();a++)
      in_host_ptr = unflatten(graph.nodes[n].attributes[a], in_host_ptr);
  
  for(int e=0;e<graph.edges.size();e++)
    for(int a=0;a<graph.edges[e].attributes.size();a++)
      in_host_ptr = unflatten(graph.edges[e].attributes[a], in_host_ptr);

  in_host_ptr = unflatten(graph.global, in_host_ptr);
  return in_host_ptr;
}
template<typename FloatType>
void unflatten(Graph<FloatType> &graph, const Vector<FloatType> &in){
  int g_size = flatSize(graph);
  assert(in.size(0)==g_size);

  autoView(in_v, in, HostRead);
  unflatten(graph, in_v.data());
}
