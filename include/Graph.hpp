#pragma once
#include <Tensors.hpp>
#include <numeric>

template<typename FloatType>
struct AttributedGraphElement{
  std::vector< Matrix<FloatType> > attributes; /**< Attributes, indexed as [attrib_type_idx][attrib_idx, batch_idx] */

  AttributedGraphElement<FloatType> & operator+=(const AttributedGraphElement<FloatType> &r);

  /**
   * @brief Zero-initialize the attributes according to the provided sizes and batch size
   */
  void initialize(const std::vector<int> &attr_sizes, int batch_size);

  /**
   * @brief Return the array of attribute sizes
   */

  std::vector<int> getAttributeSizes() const;
  
  /**
   * @brief Insert unbatched attribute information into this batched object
   * @param from The unbatched node
   * @param bidx The batch index
   */
  void insertBatch(const AttributedGraphElement<FloatType> &from, int bidx);

  /**
   * @brief Zero the attributes
   */
  void setZero();

  /**
   * @brief Return the sum of the dimension of all attributes
   */
  int totalAttribSize() const;
};

/**
 * @brief A graph node
 */
template<typename FloatType>
struct Node: public AttributedGraphElement<FloatType>{
};

/**
 * @brief A graph edge
 */
template<typename FloatType>
struct Edge: public AttributedGraphElement<FloatType>{
  int send_node;
  int recv_node;

  Edge<FloatType> & operator+=(const Edge<FloatType> &r);
  
  /**
   * @brief Insert unbatched edge information into this batched object
   * @param from The unbatched edge
   * @param bidx The batch index
   */
  void insertBatch(const Edge<FloatType> &from, int bidx);

  /**
   * @brief Zero-initialize the edge information and attributes according to the provided sizes and batch size
   */
  void initialize(int send_node, int recv_node, const std::vector<int> &attr_sizes, int batch_size);
};

/**
 * @brief Graph global attributes
 */
template<typename FloatType>
struct GlobalAttributes: public AttributedGraphElement<FloatType>{
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
  std::vector<Node<FloatType> > nodes; /**< Graph nodex */
  std::vector<Edge<FloatType> > edges; /**< Graph edges */
  GlobalAttributes<FloatType> global; /**< Graph global attributes*/

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
   * @brief Apply a lambda operation to all attributes. The function signature should be [](Matrix<FloatType> &attr){ ... }
   */
  template<typename Action>
  void applyToAllAttributes(const Action &act);

  Graph<FloatType> & operator+=(const Graph<FloatType> &r);

  /**
   * @brief Insert unbatched graph information into this batched object
   * @param from The unbatched graph
   * @param bidx The batch index
   */
  void insertBatch(const Graph<FloatType> &from, int bidx);
};




/**
 * @brief Insert unbatched attribute information into a batched attribute
 * @param into The batched attribute
 * @param from The unbatched attribute
 * @param bidx The batch index
 */
template<typename FloatType>
void batchInsertAttrib(Matrix<FloatType> &into, const Matrix<FloatType> &from, int bidx);

/**
 * @brief "Stack" an attribute matrix into the last two dimensions of an output Tensor, with a *device* pointer to the running current offset provided and output
 */
template<typename FloatType>
FloatType* stackAttr(FloatType *to_device, const Matrix<FloatType> &attr);

template<typename FloatType>
FloatType* stackAttr(FloatType *to_device, const AttributedGraphElement<FloatType> &elem);

/**
 * @brief "Unstack" an attribute matrix from the last two dimensions of an input Tensor, adding the result to the existing output, with a *device* pointer to the running current offset provided and output
 */
template<typename FloatType>
FloatType const* unstackAttrAdd(Matrix<FloatType> &attr, FloatType const* from_device);

template<typename FloatType>
FloatType const* unstackAttrAdd(AttributedGraphElement<FloatType> &elem, FloatType const* from_device);

/**
 * @brief "Unstack" an attribute matrix from the last two dimensions of an input Tensor, with a *device* pointer to the running current offset provided and output
 */
template<typename FloatType>
FloatType const* unstackAttr(Matrix<FloatType> &attr, FloatType const* from_device);

template<typename FloatType>
FloatType const* unstackAttr(AttributedGraphElement<FloatType> &elem, FloatType const* from_device);

/** 
 * @brief Support for obtaining the flattened size of a Graph and composite objects
 */
template<typename FloatType>
int flatSize(const Matrix<FloatType> &m);

template<typename FloatType>
int flatSize(const AttributedGraphElement<FloatType> &attr);

template<typename FloatType>
int flatSize(const Graph<FloatType> &g);

/**
 * @brief Flatten a graph element, outputing to the provided *host* pointer, and returning a pointer to the next array location
 */
template<typename FloatType>
FloatType* flatten(FloatType *to_host_ptr, const AttributedGraphElement<FloatType> &elem);

/**
 * @brief Unflatten a graph element, inputing from the provided *host* pointer, and returning a pointer to the next array location
 */
template<typename FloatType>
FloatType const* unflatten(AttributedGraphElement<FloatType> &elem, FloatType const *in_host_ptr);

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
