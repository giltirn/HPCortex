template<typename FloatType>
AttributedGraphElements<FloatType> & AttributedGraphElements<FloatType>::operator+=(const AttributedGraphElements<FloatType> &r){    
  int nattr = attributes.size();
  assert(r.attributes.size() == nattr);
  for(int a=0;a<nattr;a++)
    attributes[a] += r.attributes[a];
  return *this;
}

template<typename FloatType>
void AttributedGraphElements<FloatType>::insertCompleteBatch(AttributedGraphElements<FloatType> const* const* from){
  int nelem = nElem();
  int batch_size = batchSize();
  
  typedef typename AttributedGraphElements<FloatType>::AttributesType AttributesType;
  ManagedArray<typename AttributesType::View> fviews(batch_size);
  {
    autoView(fviews_v,fviews,HostWrite);
    for(int b=0;b<batch_size;b++){
      assert(from[b]->attributes.size() == attributes.size());
      for(int a=0;a<nAttrib();a++){
	int isize = attribSize(a);
	assert(from[b]->attributes[a].size(0)==nelem && from[b]->attributes[a].size(1)==isize && from[b]->attributes[a].size(2)==1);
      }
      fviews_v[b] = from[b]->attributes.view(DeviceRead);
    }
  }

  {
    autoView(fviews_v,fviews,DeviceRead);
    autoView(out_attr_v,this->attributes,DeviceWrite);
  
    for(int a=0;a<nAttrib();a++){
      int isize = attribSize(a);

      constexpr int iblocksize = 16;
      int iblocks = (isize + iblocksize - 1)/iblocksize;
    
      constexpr int bblocksize = 16;
      int bblocks = (batch_size + bblocksize - 1)/bblocksize;
    
      {
	accelerator_for_4d_gen(1,3, shm( (iblocksize+1)*bblocksize*sizeof(FloatType) ), t, iblocksize,  bi, iblocks, bblock, bblocks, e, nelem, {
	    FloatType *bstore = (FloatType*)shared;
	    int boff = bblock*bblocksize;
	    int bblocksize_actual = min(bblocksize, batch_size-boff);
	  
	    int ioff = bi*iblocksize;
	    int iblocksize_actual = min(iblocksize, isize-ioff);
	  
	    //parallel load of iblocksize from input for fixed b
	    {
	      int ii=t;
	      for(int bb=0;bb<bblocksize_actual;bb++)
		if(ii < iblocksize_actual) bstore[ii + (iblocksize+1)*bb] = fviews_v[boff + bb][a].data()[ii + ioff + isize * e];
	    }
	    acceleratorSynchronizeBlock();

	    //parallel write bblocksize consecutive elements into output for fixed i
	    for(int ii=0;ii<iblocksize_actual;ii++){
	      int i=ii+ioff;
	      int bb=t;
	      while(bb < bblocksize_actual){
		out_attr_v[a].data()[ bb + boff + batch_size*(i + isize*e) ] = bstore[ii + (iblocksize+1)*bb];
		bb += iblocksize;
	      }
	    }
	  });
      }
    }
        
    {
      autoView(fviews_v,fviews,HostRead);
      for(int b=0;b<batch_size;b++) fviews_v[b].free();
    }
  }//attributes
}

template<typename FloatType>
void AttributedGraphElements<FloatType>::initialize(int nelem, const std::vector<int> &attr_sizes, int batch_size){
  attributes.resize(attr_sizes.size());
  for(int a=0;a<attr_sizes.size();a++)
    attributes[a] = Tensor<FloatType,3>(nelem,attr_sizes[a], batch_size, FloatType(0.));
}

template<typename FloatType>
std::vector<int> AttributedGraphElements<FloatType>::getAttributeSizes() const{
  std::vector<int> out(attributes.size());
  for(int a=0;a<attributes.size();a++)
    out[a] = attributes[a].size(1);
  return out;
}

template<typename FloatType>
void AttributedGraphElements<FloatType>::setZero(){
  for(int i=0;i<attributes.size();i++){
    autoView(attr_v, attributes[i], DeviceWrite);
    acceleratorMemSet(attr_v.data(),0,attr_v.data_len() * sizeof(FloatType));
  }
}

template<typename FloatType>
int AttributedGraphElements<FloatType>::totalAttribSize() const{
  int out = 0;
  for(int a=0;a<attributes.size();a++)
    out += attributes[a].size(1);
  return out;
}

template<typename FloatType>
Edges<FloatType> & Edges<FloatType>::operator+=(const Edges<FloatType> &r){
  assert(r.edge_map.size() == edge_map.size());
  for(int e=0;e<edge_map.size();e++) assert(r.edge_map[e] == edge_map[e]);  
  this->AttributedGraphElements<FloatType>::operator+=(r);
  return *this;
}

template<typename FloatType>
void Edges<FloatType>::initialize(const std::vector<std::pair<int,int> > &_edge_map, const std::vector<int> &attr_sizes, int batch_size){
  this->edge_map = _edge_map;
  this->AttributedGraphElements<FloatType>::initialize(edge_map.size(), attr_sizes, batch_size);
}

template<typename FloatType>
Graph<FloatType>::Graph(const GraphInitialize &init){
  nodes.initialize(init.nnode, init.node_attr_sizes, init.batch_size);
  edges.initialize(init.edge_map, init.edge_attr_sizes, init.batch_size);
  global.initialize(1, init.global_attr_sizes, init.batch_size);
}

template<typename FloatType>
GraphInitialize Graph<FloatType>::getInitializer() const{
  GraphInitialize out;
  out.nnode = nodes.nElem();
  out.node_attr_sizes = nodes.getAttributeSizes();
  out.edge_attr_sizes = edges.getAttributeSizes();
  out.edge_map = edges.edge_map;
  out.global_attr_sizes = global.getAttributeSizes();
  out.batch_size = global.batchSize();
  return out;
}

template<typename FloatType>
template<typename Action>
void Graph<FloatType>::applyToAllAttributes(const Action &act){
  for(int i=0;i<nodes.attributes.size();i++)
    act(nodes.attributes[i]);
  for(int i=0;i<edges.attributes.size();i++)
    act(edges.attributes[i]);
  for(int i=0;i<global.attributes.size();i++)
    act(global.attributes[i]);
}
template<typename FloatType>
Graph<FloatType> & Graph<FloatType>::operator+=(const Graph<FloatType> &r){
  nodes += r.nodes;
  edges += r.edges;
  global += r.global;
  return *this;
}

template<typename FloatType>
Graph<FloatType> Graph<FloatType>::operator+(const Graph<FloatType> &r) const{
  Graph<FloatType> out(*this); out += r; return out;
}

template<typename FloatType>
void Graph<FloatType>::insertCompleteBatch(Graph<FloatType> const* const* from){
  int batch_size = nodes.batchSize();
  for(int b=0;b<batch_size;b++)   assert(from[b]->nodes.nElem() == nodes.nElem() && from[b]->edges.nElem() == edges.nElem());
  std::vector<AttributedGraphElements<FloatType> const* > ptrs(batch_size);  
  {
    for(int b=0;b<batch_size;b++) ptrs[b] = &from[b]->nodes;
    nodes.insertCompleteBatch(ptrs.data());    
  }
  {
    int nedge = edges.nElem();    
    for(int b=0;b<batch_size;b++){
      for(int e=0;e<nedge;e++)
	assert(from[b]->edges.edge_map[e] == edges.edge_map[e]);	
      ptrs[b] = &from[b]->edges;
    }
    edges.insertCompleteBatch(ptrs.data());    
  }
  {
    for(int b=0;b<batch_size;b++) ptrs[b] = &from[b]->global;
    global.insertCompleteBatch(ptrs.data());
  }
}

template<typename FloatType>
void stackAttr(Tensor<FloatType,3> &to,
	       const AttributedGraphElements<FloatType> &from
	       ){  
  autoView(from_v,from.attributes,DeviceRead);
  autoView(to_v,to,DeviceWrite);
  int nelem = from.nElem();
  int nattrib = from.nAttrib();
  int batch_size = from.batchSize();
  assert(to.size(0) == nelem);  
  accelerator_for_3d_gen(1,2,normal(),b,batch_size, e, nelem, a, nattrib, {
      int off = 0;
      for(int aa=0;aa<a;aa++)
	off += from_v[aa].size(1);
      auto av = from_v[a];
      for(int i=0;i<av.size(1);i++)
	to_v(e, off+i, b) = av(e, i, b);
    });
};
template<typename FloatType>
void unstackAttr(AttributedGraphElements<FloatType> &to,
		 const Tensor<FloatType,3> &from
		 ){  
  autoView(to_v,to.attributes,DeviceWrite);
  autoView(from_v,from,DeviceRead);
  int nelem = to.nElem();
  int nattrib = to.nAttrib();
  int batch_size = to.batchSize();
  assert(from.size(0) == nelem);  
  accelerator_for_3d_gen(1,2,normal(),b,batch_size, e, nelem, a, nattrib, {
      int off = 0;
      for(int aa=0;aa<a;aa++)
	off += to_v[aa].size(1);
      auto av = to_v[a];
      for(int i=0;i<av.size(1);i++)
	av(e, i, b) = from_v(e, off+i, b);
    });
};  

template<typename FloatType>
void stackAttrSingleElem(Tensor<FloatType,2> &to,
			 const AttributedGraphElements<FloatType> &from
			 ){
  
  autoView(from_v,from.attributes,DeviceRead);
  autoView(to_v,to,DeviceWrite);
  assert(from.nElem()==1);
  int nattrib = from.nAttrib();
  int batch_size = from.batchSize();
  accelerator_for_2d_gen(1,1,normal(),b,batch_size, a, nattrib, {
      int off = 0;
      for(int aa=0;aa<a;aa++)
	off += from_v[aa].size(1);
      auto av = from_v[a];
      for(int i=0;i<av.size(1);i++)
	to_v(off+i, b) = av(0, i, b);
    });
};
template<typename FloatType>
void unstackAttrSingleElem(AttributedGraphElements<FloatType> &to,
			   const Tensor<FloatType,2> &from
			   ){
  
  autoView(to_v,to.attributes,DeviceWrite);
  autoView(from_v,from,DeviceRead);
  assert(to.nElem()==1);
  int nattrib = to.nAttrib();
  int batch_size = to.batchSize();
  accelerator_for_2d_gen(1,1,normal(),b,batch_size, a, nattrib, {
      int off = 0;
      for(int aa=0;aa<a;aa++)
	off += to_v[aa].size(1);
      auto av = to_v[a];
      for(int i=0;i<av.size(1);i++)
	av(0, i, b) = from_v(off+i, b);
    });
};   


template<typename FloatType>
void stackAttr(Tensor<FloatType,3> &to,
	       const Graph<FloatType> &from,
	       const Matrix<elemCopyTemplate> &copy_template //[out_elem, copy]
	       ){
  int out_nelem = copy_template.size(0);
  assert(to.size(0) == out_nelem);
  int batch_size = to.size(2);
  int ncopy = copy_template.size(1);

  autoView(copy_template_v, copy_template, DeviceRead);
  autoView(to_v, to, DeviceWrite);
  typedef typename AttributedGraphElements<FloatType>::AttributesType AttributesType;
  
  ManagedArray<typename AttributesType::View> from_attr_views(3);
  {
    autoView(from_attr_views_v,from_attr_views,HostWrite);
    from_attr_views_v[0] = from.edges.attributes.view(DeviceRead);
    from_attr_views_v[1] = from.nodes.attributes.view(DeviceRead);
    from_attr_views_v[2] = from.global.attributes.view(DeviceRead);
  }
  autoView(from_attr_views_v, from_attr_views, DeviceRead);
  
  accelerator_for_3d_gen(1,2,normal(),b,batch_size, e, out_nelem, c, ncopy, {
      elemCopyTemplate templ = copy_template_v(e,c);
      FloatType *to = &to_v(e, templ.stacked_offset, 0);

      auto fv = from_attr_views_v[(int)templ.gelem_type][templ.gelem_attrib];
      int attr_size = fv.size(1);
      FloatType const* from = &fv(templ.gelem_elem,0,0);
      for(int i=0;i<attr_size;i++)
	to[b + batch_size*i] = from[b + batch_size*i];
    });

  {
    autoView(from_attr_views_v,from_attr_views,HostRead);
    from_attr_views_v[0].free();
    from_attr_views_v[1].free();
    from_attr_views_v[2].free();
  }
}

template<typename FloatType>
void unstackAttrAdd(Graph<FloatType> &to,
		    const Tensor<FloatType,3> &from,
		    const Matrix<elemCopyTemplate> &copy_template //[out_elem, copy]
			 ){
  int in_nelem = copy_template.size(0);
  assert(from.size(0) == in_nelem);
  int batch_size = from.size(2);
  int ncopy = copy_template.size(1);

  autoView(copy_template_v, copy_template, DeviceRead);
  autoView(from_v, from, DeviceRead);
  typedef typename AttributedGraphElements<FloatType>::AttributesType AttributesType;
  
  ManagedArray<typename AttributesType::View> to_attr_views(3);
  {
    autoView(to_attr_views_v,to_attr_views,HostWrite);
    to_attr_views_v[0] = to.edges.attributes.view(DeviceReadWrite);
    to_attr_views_v[1] = to.nodes.attributes.view(DeviceReadWrite);
    to_attr_views_v[2] = to.global.attributes.view(DeviceReadWrite);
  }
  autoView(to_attr_views_v, to_attr_views, DeviceRead);
  
  accelerator_for_3d_gen(1,2,normal(),b,batch_size, e, in_nelem, c, ncopy, {
      elemCopyTemplate templ = copy_template_v(e,c);
      auto fv = to_attr_views_v[(int)templ.gelem_type][templ.gelem_attrib];
      int attr_size = fv.size(1);
      FloatType * to = &fv(templ.gelem_elem,0,0);
      FloatType const* from = &from_v(e, templ.stacked_offset, 0);
      for(int i=0;i<attr_size;i++)
	atomicAdd( to + b + batch_size*i, from[b + batch_size*i]);
    });

  {
    autoView(to_attr_views_v,to_attr_views,HostRead);
    to_attr_views_v[0].free();
    to_attr_views_v[1].free();
    to_attr_views_v[2].free();
  }
}



template<typename FloatType>
void stackAttr(Tensor<FloatType,2> &to,
	       const Graph<FloatType> &from,
	       const Vector<elemCopyTemplate> &copy_template //[copy]
	       ){
  int batch_size = to.size(1);
  int ncopy = copy_template.size(0);

  autoView(copy_template_v, copy_template, DeviceRead);
  autoView(to_v, to, DeviceWrite);
  typedef typename AttributedGraphElements<FloatType>::AttributesType AttributesType;
  
  ManagedArray<typename AttributesType::View> from_attr_views(3);
  {
    autoView(from_attr_views_v,from_attr_views,HostWrite);
    from_attr_views_v[0] = from.edges.attributes.view(DeviceRead);
    from_attr_views_v[1] = from.nodes.attributes.view(DeviceRead);
    from_attr_views_v[2] = from.global.attributes.view(DeviceRead);
  }
  autoView(from_attr_views_v, from_attr_views, DeviceRead);
  
  accelerator_for_2d_gen(1,1,normal(),b,batch_size, c, ncopy, {
      elemCopyTemplate templ = copy_template_v(c);
      FloatType *to = &to_v(templ.stacked_offset, 0);

      auto fv = from_attr_views_v[(int)templ.gelem_type][templ.gelem_attrib];
      int attr_size = fv.size(1);
      FloatType const* from = &fv(templ.gelem_elem,0,0);
      for(int i=0;i<attr_size;i++)
	to[b + batch_size*i] = from[b + batch_size*i];
    });

  {
    autoView(from_attr_views_v,from_attr_views,HostRead);
    from_attr_views_v[0].free();
    from_attr_views_v[1].free();
    from_attr_views_v[2].free();
  }
}


template<typename FloatType>
void unstackAttrAdd(Graph<FloatType> &to,
		    const Tensor<FloatType,2> &from,
		    const Vector<elemCopyTemplate> &copy_template //[out_elem, copy]
			 ){
  int batch_size = from.size(1);
  int ncopy = copy_template.size(0);

  autoView(copy_template_v, copy_template, DeviceRead);
  autoView(from_v, from, DeviceRead);
  typedef typename AttributedGraphElements<FloatType>::AttributesType AttributesType;
  
  ManagedArray<typename AttributesType::View> to_attr_views(3);
  {
    autoView(to_attr_views_v,to_attr_views,HostWrite);
    to_attr_views_v[0] = to.edges.attributes.view(DeviceReadWrite);
    to_attr_views_v[1] = to.nodes.attributes.view(DeviceReadWrite);
    to_attr_views_v[2] = to.global.attributes.view(DeviceReadWrite);
  }
  autoView(to_attr_views_v, to_attr_views, DeviceRead);
  
  accelerator_for_2d_gen(1,1,normal(),b,batch_size, c, ncopy, {
      elemCopyTemplate templ = copy_template_v(c);
      auto fv = to_attr_views_v[(int)templ.gelem_type][templ.gelem_attrib];
      int attr_size = fv.size(1);
      FloatType * to = &fv(templ.gelem_elem,0,0);
      FloatType const* from = &from_v(templ.stacked_offset, 0);
      for(int i=0;i<attr_size;i++)
	atomicAdd( to + b + batch_size*i, from[b + batch_size*i]);
    });

  {
    autoView(to_attr_views_v,to_attr_views,HostRead);
    to_attr_views_v[0].free();
    to_attr_views_v[1].free();
    to_attr_views_v[2].free();
  }
}

template<typename FloatType>
int flatSize(const Tensor<FloatType,3> &m){
  return m.size(0)*m.size(1)*m.size(2);
}

template<typename FloatType>
int flatSize(const AttributedGraphElements<FloatType> &elem){
  int out = 0;
  for(int i=0; i<elem.nAttrib();i++)
    out += flatSize(elem.attributes[i]);
  return out;
}

template<typename FloatType>
int flatSize(const Graph<FloatType> &g){
  return flatSize(g.nodes) + flatSize(g.edges) + flatSize(g.global);
}

template<typename FloatType>
FloatType* flatten(FloatType *to_host_ptr, const AttributedGraphElements<FloatType> &elem){
  for(int i=0;i<elem.nAttrib();i++)
    to_host_ptr = flatten(to_host_ptr, elem.attributes[i]);
  return to_host_ptr;
}
template<typename FloatType>
FloatType const* unflatten(AttributedGraphElements<FloatType> &elem, FloatType const *in_host_ptr){
  for(int i=0;i<elem.nAttrib();i++)
    in_host_ptr = unflatten(elem.attributes[i], in_host_ptr);
  return in_host_ptr;
}



template<typename FloatType>
FloatType* flatten(FloatType *to_host_ptr, const Graph<FloatType> &graph){
  to_host_ptr = flatten(to_host_ptr, graph.nodes);
  to_host_ptr = flatten(to_host_ptr, graph.edges);
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
  in_host_ptr = unflatten(graph.nodes, in_host_ptr);
  in_host_ptr = unflatten(graph.edges, in_host_ptr);
  return unflatten(graph.global, in_host_ptr);
}
template<typename FloatType>
void unflatten(Graph<FloatType> &graph, const Vector<FloatType> &in){
  int g_size = flatSize(graph);
  assert(in.size(0)==g_size);

  autoView(in_v, in, HostRead);
  unflatten(graph, in_v.data());
}
