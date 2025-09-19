#pragma once
#include <Tensors.hpp>
#include <numeric>

template<typename FloatType>
struct AttributedGraphElements{
  typedef ManagedTypeArray< Tensor<FloatType,3> > AttributesType;
  AttributesType attributes; /**< Attributes, indexed as [attrib_type_idx][elem_idx,attrib_idx, batch_idx] */

  AttributedGraphElements(){}
  
  AttributedGraphElements<FloatType> & operator+=(const AttributedGraphElements<FloatType> &r);

  /**
   * @brief Zero-initialize the attributes according to the provided sizes and batch size
   */
  void initialize(int nelem, const std::vector<int> &attr_sizes, int batch_size);

  /**
   * @brief Return the array of attribute sizes
   */
  std::vector<int> getAttributeSizes() const;
  
  /**
   * @brief Insert unbatched attribute information into this batched object for all batch indices
   * @param from An array of pointers to unbatched elements of size batch_size
   * 
   * Much faster than using insertBatch repeatedly
   */
  void insertCompleteBatch(AttributedGraphElements<FloatType> const* const* from);
  
  /**
   * @brief Zero the attributes
   */
  void setZero();

  /**
   * @brief Return the sum of the dimension of all attributes
   */
  int totalAttribSize() const;

  /**
   * @brief Return the vector size of an attribute
   */
  int attribSize(int attrib) const{ return attributes[attrib].size(1); }

  /**
   * @brief Return the number of graph elements
   */
  int nElem() const{ return attributes[0].size(0); }

  /**
   * @brief Return the number of attributes
   */
  int nAttrib() const{ return attributes.size(); }

  /**
   * @brief Return the batch size
   */  
  int batchSize() const{ return attributes[0].size(2); }
};

/**
 * @brief A collection of graph nodes
 */
template<typename FloatType>
struct Nodes: public AttributedGraphElements<FloatType>{
};

/**
 * @brief A collection of graph edges
 */
template<typename FloatType>
struct Edges: public AttributedGraphElements<FloatType>{
  std::vector<std::pair<int,int> > edge_map;  /**< The edge connectivity as [send, recv]*/

  Edges<FloatType> & operator+=(const Edges<FloatType> &r);
  
  /**
   * @brief Zero-initialize the edge information and attributes according to the provided sizes and batch size
   */
  void initialize(const std::vector<std::pair<int,int> > &_edge_map, const std::vector<int> &attr_sizes, int batch_size);

  /**
   * @brief Return the send node index associated with the given edge
   */
  int sendNode(int edge) const{ return edge_map[edge].first; }

  /**
   * @brief Return the receive node index associated with the given edge
   */
  int recvNode(int edge) const{ return edge_map[edge].second; }
  
};

/**
 * @brief Graph global attributes
 */
template<typename FloatType>
struct Globals: public AttributedGraphElements<FloatType>{
};


/**
 * @brief A structure for initializing graphs with a specific connectivity
 */
struct GraphInitialize{
  int nnode; /**< Number of nodes */
  
  std::vector<int> node_attr_sizes;  /**< The dimensions of each of the node's attributes with the size of the array specifying the number of attributes */ 
  std::vector<int> edge_attr_sizes;  /**< The dimensions of each of the edge's attributes with the size of the array specifying the number of attributes */ 
  std::vector<std::pair<int,int> > edge_map; /**< The edge connectivity as [send, recv]*/
  
  std::vector<int> global_attr_sizes; /**< The dimensions of the global attributes */
  int batch_size; /**< The batch size*/

  bool operator==(const GraphInitialize &r) const;

  static inline int totalAttribSize(const std::vector<int> &v){ return std::accumulate(v.begin(),v.end(),0); }
};

/**
 * @brief A directed graph
 */
template<typename FloatType>
struct Graph{
  Nodes<FloatType> nodes; /**< Graph nodes */
  Edges<FloatType> edges; /**< Graph edges */
  Globals<FloatType> global; /**< Graph global attributes*/

  Graph(){}

  /**
   * @brief Zero-initialize a graph with a specific layout/connectivity
   */
  Graph(const GraphInitialize &init);

  /** 
   * @brief Return a structure allowing zero initialization of a graph with the same layout/connectivity
   */
  GraphInitialize getInitializer() const;

  /**
   * @brief Apply a lambda operation to all attributes. The function signature should be [](Tensor<FloatType,3> &attr){ ... }
   */
  template<typename Action>
  void applyToAllAttributes(const Action &act);

  Graph<FloatType> & operator+=(const Graph<FloatType> &r);

  Graph<FloatType> operator+(const Graph<FloatType> &r) const;
  
  /**
   * @brief Insert unbatched graph information into this batched object for all batch indices
   * @param from An array of pointers to unbatched graphs of size batch_size
   */
  void insertCompleteBatch(Graph<FloatType> const* const* from);
};

/**
 * @brief Enum for graph components
 */
enum class GraphElementType { Edges=0, Nodes=1, Global=2 };

/**
 * @brief Depending on the reference type of 'from', either copy or move a particular graph element type into the output 'to'
 */
template<typename FloatType>
void copyOrMoveGraphElement(Graph<FloatType> &to, const Graph<FloatType> &from, const GraphElementType type);

template<typename FloatType>
void copyOrMoveGraphElement(Graph<FloatType> &to, Graph<FloatType> &&from, const GraphElementType type);

/**
 * @brief Information struct describing copies between graph elements and stacked attribute tensors
 */
struct elemCopyTemplate{
GraphElementType gelem_type; /**< Which of the AttributedGraphElements to copy from/to*/
  int gelem_attrib; /**< Attribute index */
  int gelem_elem; /**< Element index */
  int stacked_offset; /**< Offset within stacked output dimension*/
  
  elemCopyTemplate(GraphElementType gelem_type, int gelem_attrib, int gelem_elem, int stacked_offset): gelem_type(gelem_type), gelem_attrib(gelem_attrib), gelem_elem(gelem_elem), stacked_offset(stacked_offset){}  
};

/**
 * @brief "Stack" a graph's attributes associated with a specific element type into the last two dimensions of a 3D output Tensor with indices [out_elem, stacked_attrib_idx, batch_idx]. The source attributes are specified by a template
 */
template<typename FloatType>
void stackAttr(Tensor<FloatType,3> &to,
	       const AttributedGraphElements<FloatType> &from
	       );
/**
 * @brief The inverse operation to the above
 */
template<typename FloatType>
void unstackAttr(AttributedGraphElements<FloatType> &to,
		      const Tensor<FloatType,3> &from
		 );

/**
 * @brief Specifically for AttributedGraphElements instances with one element (e.g. global attributes), stack its attributes into a 2D output Tensor with indices [stacked_attrib_idx, batch_idx]
 */
template<typename FloatType>
void stackAttrSingleElem(Tensor<FloatType,2> &to,
			 const AttributedGraphElements<FloatType> &from
			 );
/**
 * @brief The inverse operation to the above.
 */
template<typename FloatType>
void unstackAttrSingleElem(AttributedGraphElements<FloatType> &to,
			   const Tensor<FloatType,2> &from
			   );

/**
 * @brief "Stack" a graph's attributes from arbitrary element types into the last two dimensions of a 3D output Tensor with indices [out_elem, stacked_attrib_idx, batch_idx]. 
 The source attributes are specified by a template matrix providing an elemCopyTemplate indexed by the output element index and the copies associated with that element
 */
template<typename FloatType>
void stackAttr(Tensor<FloatType,3> &to,
	       const Graph<FloatType> &from,
	       const Matrix<elemCopyTemplate> &copy_template //[out_elem, copy]
	       );
/**
 * @brief The inverse operation to the above. As multiple stacked outputs can be associated with a single input element (e.g. one node can contribute to the updates of multiple edges), the inverse contributions are summed into the output.
 */
template<typename FloatType>
void unstackAttrAdd(Graph<FloatType> &to,
		    const Tensor<FloatType,3> &from,
		    const Matrix<elemCopyTemplate> &copy_template //[out_elem, copy]
		    );
  
/**
 * @brief "Stack" a graph's attributes from arbitrary element types into a 2D output Tensor with indices [stacked_attrib_idx, batch_idx]. 
 The source attributes are specified by a template providing a list of copies
 */
template<typename FloatType>
void stackAttr(Tensor<FloatType,2> &to,
	       const Graph<FloatType> &from,
	       const Vector<elemCopyTemplate> &copy_template //[copy]
	       );
/**
 * @brief The inverse operation to the above. As multiple stacked outputs can be associated with a single input element (e.g. one node can contribute to the updates of multiple edges), the inverse contributions are summed into the output.
 */
template<typename FloatType>
void unstackAttrAdd(Graph<FloatType> &to,
		    const Tensor<FloatType,2> &from,
		    const Vector<elemCopyTemplate> &copy_template //[copy]
		    );

/** 
 * @brief Support for obtaining the flattened size of a Graph and composite objects
 */
template<typename FloatType>
int flatSize(const Tensor<FloatType,3> &m);

template<typename FloatType>
int flatSize(const AttributedGraphElements<FloatType> &elem);

template<typename FloatType>
int flatSize(const Graph<FloatType> &g);

/**
 * @brief Flatten a graph element, outputing to the provided *host* pointer, and returning a pointer to the next array location
 */
template<typename FloatType>
FloatType* flatten(FloatType *to_host_ptr, const AttributedGraphElements<FloatType> &elem);

/**
 * @brief Unflatten a graph element, inputing from the provided *host* pointer, and returning a pointer to the next array location
 */
template<typename FloatType>
FloatType const* unflatten(AttributedGraphElements<FloatType> &elem, FloatType const *in_host_ptr);

/**
 * @brief Flatten a graph, outputing to the provided *host* pointer, and returning a pointer to the next array location
 */
template<typename FloatType>
FloatType* flatten(FloatType *to_host_ptr, const Graph<FloatType> &graph);

/**
 * @brief Flatten a graph
 */
template<typename FloatType>
Vector<FloatType> flatten(const Graph<FloatType> &graph);

/**
 * @brief Unflatten a graph, inputing from the provided *host* pointer, and returning a pointer to the next array location
 */
template<typename FloatType>
FloatType const* unflatten(Graph<FloatType> &graph, FloatType const *in_host_ptr);

/**
 * @brief Unflatten a graph. The output must be pre-initialized to the appropriate size
 */
template<typename FloatType>
void unflatten(Graph<FloatType> &graph, const Vector<FloatType> &in);


#include "implementation/Graph.tcc"
