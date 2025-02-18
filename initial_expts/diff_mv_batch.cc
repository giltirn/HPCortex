#include<iostream>
#include<cmath>
#include<vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <array>

//ML interface with batching
struct LeafTag{};

template<size_t dim>
inline size_t tensorSize(const std::array<int,dim> &dims){
  size_t out=1;
  for(int d=0;d<dim;d++) out *= dims[d];
  return out;
}
template<size_t Dim>
inline size_t compute_off(int const* coord, int const* dims){
  size_t out = *coord++; ++dims;
  for(int i=1;i<Dim;i++) out = out * (*dims++) + (*coord++);
  return out;
}
template<size_t Dim>
inline size_t compute_off(const std::array<int,Dim> &coord, const std::array<int,Dim> &dims){
  return compute_off<Dim>(coord.data(),dims.data());
}

template<int Dim>
struct Tensor{
  std::vector<double> vals;
  int _size[Dim];
    
public:
  typedef std::array<int,Dim> Dims;
  typedef std::array<int,Dim> Coord;
  
  constexpr int dimension(){ return Dim; }
  Tensor(): _size{0}{}
  Tensor(const Dims &dims, double init): vals(tensorSize(dims),init){ memcpy(_size,dims.data(),Dim*sizeof(int));  }
  Tensor(const Dims &dims, const std::vector<double> &init_vals): vals(init_vals){
    memcpy(_size,dims.data(),Dim*sizeof(int));
    assert(tensorSize(dims) == init_vals.size());
  }  
  inline double & operator()(const Coord &coord){ return vals[compute_off<Dim>(coord.data(), size)]; }
  inline double operator()(const Coord &coord) const{ return vals[compute_off<Dim>(coord.data(), size)]; }

  inline int size(int i) const{ return _size[i]; }

};

struct Vector{
  std::vector<double> vals;
public:
  Vector(){}
  Vector(int size1): vals(size1){}
  Vector(int size1, double init): vals(size1, init){}
  Vector(const std::vector<double> &init_vals): vals(init_vals){}    
  
  inline double & operator()(const int i){ return vals[i]; }
  inline double operator()(const int i) const{ return vals[i]; }

  inline int size(int i) const{ return vals.size(); }
};

struct Matrix{
  std::vector<double> vals;
  int size0;
  int size1;
public:
  Matrix(): size0(0),size1(0){}
  Matrix(int size0, int size1): size0(size0), size1(size1), vals(size0*size1){}  
  Matrix(int size0, int size1, double init): size0(size0), size1(size1), vals(size0*size1,init){}
  Matrix(int size0, int size1, const std::vector<double> &init_vals): size0(size0), size1(size1), vals(init_vals){}    
  
  inline double & operator()(const int i, const int j){ return vals[j+size1*i]; }
  inline double operator()(const int i, const int j) const{ return vals[j+size1*i]; }

  inline int size(int i) const{ return i==0 ? size0 : size1; }

  void pokeColumn(int col, const Vector &data){
    assert(data.size(0) == size0);
    for(int i=0;i<size0;i++)
      this->operator()(i,col) = data(i);
  }
  Vector peekColumn(int col) const{
    Vector out(size0);
    for(int i=0;i<size0;i++) out(i)=this->operator()(i,col);
    return out;
  }
   
};

Vector operator*(const Matrix &A, const Vector &x){
  Vector out(A.size(0), 0.);
  for(int i=0;i<A.size(0);i++)
    for(int j=0;j<A.size(1);j++)
      out(i) += A(i,j) * x(j);
  return out;
}
Vector operator+(const Vector &a, const Vector &b){
  Vector out(a.size(0));
  for(int i=0;i<a.size(0);i++)
    out(i) = a(i) + b(i);
  return out;
}
Vector operator-(const Vector &a, const Vector &b){
  Vector out(a.size(0));
  for(int i=0;i<a.size(0);i++)
    out(i) = a(i) - b(i);
  return out;
}
Vector operator*(double eps, const Vector &b){
  Vector out(b.size(0));
  for(int i=0;i<b.size(0);i++)
    out(i) = eps * b(i);
}


class InputLayer{
public:
  typedef LeafTag tag;
  
  inline InputLayer(){}

  inline const Matrix &value(const Matrix &x){
    return x;
  }

  inline void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv) const{ assert(off == cost_deriv.size(0)); }
  
  inline void update(int off, const Vector &new_params){}

  inline void step(int off, const Vector &derivs, double eps){}
  
  inline int nparams(){ return 0; }

  inline void getParams(Vector &into, int off){}    
};

inline InputLayer input_layer(){ return InputLayer(); }

template<typename T>
struct LeafStore{
  T v;
  LeafStore(T && v): v(v){
    //std::cout << "STORE" << std::endl;
  }
};
template<typename T>
struct LeafRef{
  T &v;
  LeafRef(T &v): v(v){
    //std::cout << "REF" << std::endl;
  }
};

template<typename T>
struct deduceStorage{};
template<typename T>
struct deduceStorage<T&>{
  typedef LeafRef<T> type;
};

template<typename T>
struct deduceStorage<T&&>{
  typedef LeafStore<T> type;
};

#define DDST(a) typename deduceStorage<decltype(a)>::type

#define ISLEAF(a) std::is_same<typename std::decay<a>::type::tag,LeafTag>::value

class ReLU{
public: 
  inline Matrix operator()(const Matrix &x) const{
    int dim = x.size(0);
    int batch_size = x.size(1);
    Matrix out(dim,batch_size,1.0);
    //f(x)_i = max(x_i, 0)
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	if(x(i,b) <= 0.) out(i,b) = 0.;
    return out;
  }
};

class noActivation{
public:
  inline Matrix operator()(const Matrix &x) const{
    return Matrix(x.size(0),x.size(1),1.0);
  }
};

  


template<typename Store, typename ActivationFunc>
class DNNlayer{
  Matrix weights;
  Vector bias;  
  Store leaf;
  int size0;
  int size1;

  ActivationFunc activation_func;

  //Storage from last call to "value"
  Matrix leaf_val;
  Matrix activation;
public:
  typedef LeafTag tag;
  
  DNNlayer(Store &&leaf, const Matrix &weights,const Vector &bias, const ActivationFunc &activation_func):
    leaf(leaf), weights(weights), bias(bias),
    size0(weights.size(0)), size1(weights.size(1)),
    activation_func(activation_func)
  {}

  //Forward pass
  Matrix value(const Matrix &x){
    leaf_val = leaf.v.value(x);
    int batch_size = x.size(1);   
    assert(leaf_val.size(0) == size1);
    assert(leaf_val.size(1) == batch_size);

    Matrix out(size0,batch_size,0.0);

    for(int i=0;i<size0;i++){
      for(int b=0;b<batch_size;b++){
	out(i,b) = bias(i);
	for(int j=0;j<size1;j++)
	  out(i,b) += weights(i,j)* leaf_val(j,b);
      }
    }
	
    activation = activation_func(out);
    assert(activation.size(0) == size0);
    assert(activation.size(1) == batch_size);
    
    for(int i=0;i<size0;i++)
      for(int b=0;b<batch_size;b++)
	out(i,b) *= activation(i,b);    
    
    return out;
  }
 
  void deriv(Vector &cost_deriv, int off, const Matrix &above_deriv) const{
    assert(above_deriv.size(0) == size0);
    int batch_size = leaf_val.size(1);
    
    //for reverse differentiation, we pass down the derivatives with respect to our inputs
    //f(x)_i = act_i b_i + \sum_j act_i w_ij x_j
    //dcost / dx_j = \sum_i dcost/df_i df_i/dx_j
    //df_i/dx_j = act_i w_ij
    Matrix layer_deriv(size1,batch_size,0.);
    for(int j=0;j<size1;j++)
      for(int i=0;i<size0;i++)
	for(int b=0;b<batch_size;b++)
	  layer_deriv(j,b) += above_deriv(i,b) * activation(i,b) * weights(i,j);

    //Now we finish up the derivs wrt our parameters
    //df(x)_i / d w_jk = delta_ij act_j x_k
    //df(x)_i / d b_j = delta_ij act_j
    //dcost / dw_jk = \sum_i dcost/df_i df_i/dw_jk = dcost/df_j * act_j * x_k
    //dcost / db_j = \sum_i dcost/df_i df_i/db_j = dcost/df_j * act_j
    int p=off;
    for(int j=0;j<size0;j++)
      for(int k=0;k<size1;k++){
	for(int b=0;b<batch_size;b++)
	  cost_deriv(p) += above_deriv(j,b) * activation(j,b) * leaf_val(k,b); //batch reduction! (assume zero-initialized)
	++p;
      }
	
    for(int j=0;j<size0;j++){
      for(int b=0;b<batch_size;b++)
	cost_deriv(p) += above_deriv(j,b) * activation(j,b);
      ++p;
    }
    
    leaf.v.deriv(cost_deriv, p, layer_deriv);
  }

  void update(int off, const Vector &new_params){
    int p=off;
    for(int i=0;i<size0;i++)
      for(int j=0;j<size1;j++)
	weights(i,j) = new_params(p++);
    for(int i=0;i<size0;i++)
      bias(i) = new_params(p++);
    leaf.v.update(p, new_params);
  }
  void step(int off, const Vector &derivs, double eps){
    int p=off;
    for(int i=0;i<size0;i++)
      for(int j=0;j<size1;j++){
	//std::cout << "Weights " << i << " " << j << " " << weights(i,j) << " -= " << derivs(p) << "*" << eps;
	weights(i,j) -= derivs(p++) * eps;
	//std::cout << " = " <<  weights(i,j) << std::endl;
      }
    for(int i=0;i<size0;i++){
      //std::cout << "Bias " << i << " " << bias(i) << " -= " << derivs(p) << "*" << eps;
      bias(i) -= derivs(p++) * eps;
      //std::cout << " = " << bias(i) << std::endl;
    }
    leaf.v.step(p, derivs, eps);
  }

  //accumulated #params for layers here and below
  inline int nparams(){ return size0*size1 + size0 + leaf.v.nparams(); }

  //off measured from *end*, return new off
  void getParams(Vector &into, int off){
    int p = off;
    for(int i=0;i<size0;i++)
      for(int j=0;j<size1;j++)
	into(p++) = weights(i,j);
    for(int i=0;i<size0;i++)
      into(p++) = bias(i);
    leaf.v.getParams(into, p);
  }
};

template<typename U, typename ActivationFunc, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix &weights,const Vector &bias, const ActivationFunc &activation)->DNNlayer<DDST(u),ActivationFunc>{
  return DNNlayer<DDST(u),ActivationFunc>(std::forward<U>(u), weights, bias, activation);
}
template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto dnn_layer(U &&u, const Matrix &weights,const Vector &bias)->DNNlayer<DDST(u),noActivation>{
  return DNNlayer<DDST(u),noActivation>(std::forward<U>(u), weights, bias, noActivation());
}


template<typename Store>
class MSEcost{
  Store leaf;
  Matrix ypred; //dim * batch_size
  Matrix yval;
  int nparam;
public:
  MSEcost(Store &&leaf): leaf(leaf), nparam(leaf.v.nparams()){}

  double loss(const Matrix &x, const Matrix &y){
    //out = \sum_j (ypred(j) - y(j))^2/dim
    ypred = leaf.v.value(x);
    int dim = y.size(0);
    int batch_size = y.size(1);
    assert(ypred.size(0) == dim);
    assert(ypred.size(1) == batch_size);
    
    yval = y;
    double out = 0.;
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	out += pow(ypred(i,b) - y(i,b),2);
    out /= (dim * batch_size);
    return out;
  }
  Matrix predict(const Matrix &x){
    return leaf.v.value(x);
  }
 
  Vector deriv() const{
    //for reverse differentiation, we pass down the derivatives with respect to our inputs
    //dout / dparam(i) = \sum_j 2*(ypred(j) - y(j)) * dypred(j)/dparam(i)

    //dout / dypred(i) = 2*(ypred(i) - y(i)) /dim
    int dim = yval.size(0);
    int batch_size = yval.size(1);
    
    Matrix layer_deriv(dim,batch_size);
    for(int i=0;i<dim;i++)
      for(int b=0;b<batch_size;b++)
	layer_deriv(i,b) = 2*(ypred(i,b) - yval(i,b)) / (dim*batch_size);

    Vector cost_deriv(nparam,0.);    //zero initialize
    leaf.v.deriv(cost_deriv, 0, layer_deriv);
    return cost_deriv;
  }
  
  void update(const Vector &new_params){
    leaf.v.update(0, new_params);
  }
  void step(const Vector &derivs, double eps){
    leaf.v.step(0,derivs,eps);
  }
  int nparams(){ return nparam; }

  Vector getParams(){
    Vector out(nparams());
    leaf.v.getParams(out,0);
    return out;
  }
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto mse_cost(U &&u)->MSEcost<DDST(u)>{
  return MSEcost<DDST(u)>(std::forward<U>(u));
}

struct XYpair{
  Matrix x;
  Matrix y;
};


template<typename T, typename LRscheduler>
void optimizeGradientDescent(T &model, const std::vector<XYpair> &data, const LRscheduler &lr, int nepoch){
  std::default_random_engine gen(1234);
  std::uniform_int_distribution<int> dist(0,data.size());

  int ndata = data.size();
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;
  
  for(int epoch=0;epoch<nepoch;epoch++){
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );
    double eps = lr(epoch);
    std::cout << "Epoch " << epoch << " learning rate " << eps << std::endl;
    
    for(int ii=0;ii<ndata;ii++){
      int i = didx[ii];
      double loss = model.loss(data[i].x, data[i].y);
      std::cout << epoch << "-" << ii << " : "<< loss << std::endl;
      model.step( model.deriv(), eps );
    }
  }

}



struct AdamParams{ //NB, alpha comes from the learning scheduler
  double beta1;
  double beta2;
  double eps;
  AdamParams( double beta1=0.99, double beta2=0.999, double eps=1e-8): beta1(beta1), beta2(beta2), eps(eps){}
};
  
template<typename T, typename LRscheduler>
void optimizeAdam(T &model, const std::vector<XYpair> &data, const LRscheduler &lr, const AdamParams &ap, int nepoch){
  std::default_random_engine gen(1234);
  std::uniform_int_distribution<int> dist(0,data.size());

  int nparam = model.nparams();
  Vector m(nparam,0.0);
  Vector v(nparam,0.0);
  int t=0;
  
  int ndata = data.size();
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;
  
  for(int epoch=0;epoch<nepoch;epoch++){
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );
    double alpha = lr(epoch);
    std::cout << "Epoch " << epoch << " learning rate " << alpha << std::endl;
    
    for(int ii=0;ii<ndata;ii++){
      int i = didx[ii];
      double loss = model.loss(data[i].x, data[i].y);
      auto g = model.deriv();

      double delta = t>0 ? alpha * sqrt(1. - pow(ap.beta2,t))  / (1. - pow(ap.beta1,t) ) : alpha;
      for(int p=0;p<nparam;p++){
	double gp_init = g(p);
	m(p) = ap.beta1 * m(p) + (1.-ap.beta1)*g(p);
	v(p) = ap.beta2 * v(p) + (1.-ap.beta2)*pow(g(p),2);

	g(p) = m(p)/(sqrt(v(p)) + ap.eps);
	//std::cout << "p="<< p << " m=" << m(p) << " v=" << v(p) << " g:" << gp_init << "->" <<  g(p) << std::endl;
      }
      ++t;      
      std::cout << epoch << "-" << ii << " : "<< loss << " update model with step size " << delta << std::endl;
      model.step( g , delta );
    }
  }

}


//TODO: Optimizer can be separate, needs to be passed just the gradient and return an ascent vector and step size
//TODO: Consider how to distribute layers over MPI. Each rank has a batch of layers. We need to keep every rank busy
//      Need distributed vectors and operations thereon

class DecayScheduler{
  double eps;
  double decay_rate;
public:
  DecayScheduler(double eps, double decay_rate): eps(eps), decay_rate(decay_rate){}
  double operator()(const int epoch) const{ return eps * 1./(1. + decay_rate * epoch); }
};


void basicTests(){
  typedef std::vector<double> vecD;
  
  Matrix w1_init(3,2, vecD({0.1,0.2,
	                   -0.1,-0.2,
			    0.7,0.7}));
  Vector b1_init( vecD({0.5,0.7,0.9}));		    
  
  auto f = mse_cost( dnn_layer(input_layer(), w1_init, b1_init) );

  //NB batch size 2, batches in different *columns*
  Matrix x1(2,2,vecD({1.3, 0.6,
	             -0.3, -1.7}));
  
  Matrix y1(3,2,std::vector<double>({-0.5, -0.5,
	                             1.7, 1.3
				     -0.7, -0.5}));

  double expect = 0.;
  for(int i=0;i<2;i++){  
    Vector y1pred = w1_init * x1.peekColumn(i) + b1_init;
    Vector y1_b = y1.peekColumn(i);
    expect += pow(y1pred(0)-y1_b(0),2)/3. + pow(y1pred(1)-y1_b(1),2)/3. + pow(y1pred(2)-y1_b(2),2)/3.;
  }
  expect /= 2.;
    
  double got=  f.loss(x1,y1);
  std::cout << "Test loss : got " << got << " expect " << expect << std::endl;


  Vector dexpect(9);
  int p=0;
  for(int i=0;i<3;i++){
    for(int j=0;j<2;j++){
      Matrix w1_p = w1_init;
      w1_p(i,j) += 1e-7;
      auto f2 = mse_cost( dnn_layer(input_layer(), w1_p, b1_init) );
      dexpect(p++) = (f2.loss(x1,y1) - got)/1e-7;
    }
  }
  for(int i=0;i<3;i++){
    Vector b1_p = b1_init;
    b1_p(i) += 1e-7;
    auto f2 = mse_cost( dnn_layer(input_layer(), w1_init, b1_p) );
    dexpect(p++) = (f2.loss(x1,y1) - got)/1e-7;    
  }

  Vector dgot = f.deriv();
  for(int i=0;i<9;i++){
    std::cout << "Test deriv wrt param " << i <<  ": got " << dgot(i) << " expect " << dexpect(i) << std::endl;
  }
    
  //test update
  Matrix w1_new(3,2, std::vector<double>({-0.5,0.4,
					  0.8,1.2,
					  2.1,-3.0}));
  Vector b1_new( std::vector<double>({-0.5,0.7,-1.1}));	

  auto ftest = mse_cost( dnn_layer(input_layer(), w1_new, b1_new) );
  f.update(ftest.getParams());
  
  std::cout << "Update check : expect " << ftest.loss(x1,y1) << " got " <<  f.loss(x1,y1) << std::endl;

}


void testSimpleLinear(){
  //Test f(x) = 0.2*x + 0.3;

  Matrix winit(1,1,0.0);
  Vector binit(1,0.0);

  int ndata = 100;
  std::vector<XYpair> data(ndata);
  for(int i=0;i<ndata;i++){
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1

    data[i].x = Matrix(1,1,x);
    data[i].y = Matrix(1,1,0.2*x + 0.3);
  }
    
  auto model = mse_cost( dnn_layer(input_layer(), winit, binit) );
  DecayScheduler lr(0.01, 0.1);
  optimizeGradientDescent(model, data, lr, 200);

  std::cout << "Final params" << std::endl;
  Vector final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p(i) << std::endl;

}


void testOneHiddenLayer(){
  //Test f(x) = 0.2*x + 0.3;
  int nbatch = 100;
  int batch_size = 4;
  std::vector<XYpair> data(nbatch);

  int ndata = batch_size * nbatch;

  for(int i=0;i<ndata;i++){ //i = b + batch_size * B
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1

    int b = i % batch_size;
    int B = i / batch_size;
    if(b==0){
      data[B].x = Matrix(1,batch_size);
      data[B].y = Matrix(1,batch_size);
    }
    
    data[B].x(0,b) = x;
    data[B].y(0,b) = 0.2*x + 0.3;
  }

  int nhidden = 5;

  Matrix winit_out(1,nhidden,0.01);
  Matrix winit_h(nhidden,1,0.01);

  Vector binit_out(1,0.01);
  Vector binit_h(nhidden, 0.01);

  auto hidden_layer = dnn_layer(input_layer(), winit_h, binit_h, ReLU());
  auto model = mse_cost( dnn_layer(hidden_layer, winit_out, binit_out) );

  //Test derivative
  {
    Vector p = model.getParams();
    
    for(int d=1;d<5;d++){ //first 5 data
    
      double c1 = model.loss(data[d].x,data[d].y);
      Vector pd = model.deriv();
      
      auto hidden_layer2 = dnn_layer(input_layer(), winit_h, binit_h, ReLU());  
      auto model2 = mse_cost( dnn_layer(hidden_layer2, winit_out, binit_out) );

      std::cout << "Test derivs " << d << " x=" << data[d].x(0,0) << " " << data[d].x(0,1) << std::endl;
      for(int i=0;i<p.size(0);i++){
	Vector pp(p);
	pp(i) += 1e-9;
	model2.update(pp);
      
	double c2 = model2.loss(data[d].x,data[d].y);
	std::cout << i << " got " << pd(i) << " expect " << (c2-c1)/1e-9 << std::endl;
      }
    }
  }


  DecayScheduler lr(0.001, 0.1);
  AdamParams ap;
  optimizeAdam(model, data, lr, ap, 200);

  std::cout << "Final params" << std::endl;
  Vector final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p(i) << std::endl;

  std::cout << "Test on some data" << std::endl;
  for(int d=0;d<data.size();d++){ //first 5 data, batch idx 0
    auto got = model.predict(data[d].x);
    std::cout << data[d].x(0,0) << " got " << got(0,0) << " expect " << data[d].y(0,0) << std::endl;
  }

}


int main(void){
  //basicTests();
  //testSimpleLinear();
  testOneHiddenLayer();

  return 0;
}
