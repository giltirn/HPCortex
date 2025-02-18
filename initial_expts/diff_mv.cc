#include<iostream>
#include<cmath>
#include<vector>
#include <random>
#include <algorithm>
#include <cassert>

//First implementation of DNN with backwards differentiation including ADAM and gradient descent optimizers
struct LeafTag{};

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
  int size1;
  int size2;
public:
  Matrix(int size1, int size2): size1(size1), size2(size2), vals(size1*size2){}  
  Matrix(int size1, int size2, double init): size1(size1), size2(size2), vals(size1*size2,init){}
  Matrix(int size1, int size2, const std::vector<double> &init_vals): size1(size1), size2(size2), vals(init_vals){}    
  
  inline double & operator()(const int i, const int j){ return vals[j+size2*i]; }
  inline double operator()(const int i, const int j) const{ return vals[j+size2*i]; }

  inline int size(int i) const{ return i==0 ? size1 : size2; }
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
  Vector val;
public:
  typedef LeafTag tag;
  
  InputLayer(){}
  const Vector &value(const Vector &x){
    val = x;    
    return val;
  }

  void deriv(Vector &cost_deriv, int off, const Vector &above_deriv) const{ assert(off == cost_deriv.size(0)); }
  
  inline void update(int off, const Vector &new_params){}

  inline void step(int off, const Vector &derivs, double eps){}
  
  int nparams(){ return 0; }

  inline void getParams(Vector &into, int off){}    
};

InputLayer input_layer(){ return InputLayer(); }

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
  inline Vector operator()(const Vector &x) const{
    Vector out(x.size(0),1.0);
    //f(x)_i = max(x_i, 0)
    for(int i=0;i<x.size(0);i++) if(x(i) <= 0.) out(i) = 0.;
    return out;
  }
};

class noActivation{
public:
  inline Vector operator()(const Vector &x) const{
    return Vector(x.size(0),1.0);
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
  Vector leaf_val;
  Vector activation;
public:
  typedef LeafTag tag;
  
  DNNlayer(Store &&leaf, const Matrix &weights,const Vector &bias, const ActivationFunc &activation_func):
    leaf(leaf), weights(weights), bias(bias),
    size0(weights.size(0)), size1(weights.size(1)),
    activation_func(activation_func)
  {}

  //Forward pass
  Vector value(const Vector &x){
    leaf_val = leaf.v.value(x);
    assert(leaf_val.size(0) == size1);
    
    Vector out(bias);
    for(int i=0;i<size0;i++)
      for(int j=0;j<size1;j++)
	out(i) += weights(i,j)* leaf_val(j);
    
    activation = activation_func(out); assert(activation.size(0) == size0);
    for(int i=0;i<size0;i++) out(i) *= activation(i);
    
    return out;
  }
 
  void deriv(Vector &cost_deriv, int off, const Vector &above_deriv) const{
    assert(above_deriv.size(0) == size0);
    
    //for reverse differentiation, we pass down the derivatives with respect to our inputs
    //f(x)_i = act_i b_i + \sum_j act_i w_ij x_j
    //dcost / dx_j = \sum_i dcost/df_i df_i/dx_j
    //df_i/dx_j = act_i w_ij
    Vector layer_deriv(size1,0.);
    for(int j=0;j<size1;j++)    
      for(int i=0;i<size0;i++)
	layer_deriv(j) += above_deriv(i) * activation(i) * weights(i,j);

    //Now we finish up the derivs wrt our parameters
    //df(x)_i / d w_jk = delta_ij act_j x_k
    //df(x)_i / d b_j = delta_ij act_j
    //dcost / dw_jk = \sum_i dcost/df_i df_i/dw_jk = dcost/df_j * act_j * x_k
    //dcost / db_j = \sum_i dcost/df_i df_i/db_j = dcost/df_j * act_j
    int p=off;
    for(int j=0;j<size0;j++)
      for(int k=0;k<size1;k++)
	cost_deriv(p++) = above_deriv(j) * activation(j) * leaf_val(k);
    
    for(int j=0;j<size0;j++)
      cost_deriv(p++) = above_deriv(j) * activation(j);
    
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
  Vector ypred;
  Vector yval;
  int nparam;
public:
  MSEcost(Store &&leaf): leaf(leaf), nparam(leaf.v.nparams()){}

  double loss(const Vector &x, const Vector &y){
    //out = \sum_j (ypred(j) - y(j))^2/dim
    ypred = leaf.v.value(x);
    int dim = y.size(0);
    assert(ypred.size(0) == dim);
    yval = y;
    double out = 0.;
    for(int i=0;i<dim;i++)
      out += pow(ypred(i) - y(i),2);
    out /= dim;
    return out;
  }
  Vector predict(const Vector &x){
    return leaf.v.value(x);
  }
 
  Vector deriv() const{
    //for reverse differentiation, we pass down the derivatives with respect to our inputs
    //dout / dparam(i) = \sum_j 2*(ypred(j) - y(j)) * dypred(j)/dparam(i)

    //dout / dypred(i) = 2*(ypred(i) - y(i)) /dim
    int dim = yval.size(0);
    Vector layer_deriv(dim);
    for(int i=0;i<dim;i++) layer_deriv(i) = 2*(ypred(i) - yval(i)) / dim;

    Vector cost_deriv(nparam,0.);    
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
  Vector x;
  Vector y;
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
  Matrix w1_init(3,2, std::vector<double>({0.1,0.2,
					  -0.1,-0.2,
					  0.7,0.7}));
  Vector b1_init( std::vector<double>({0.5,0.7,0.9}));		    
  
  auto f = mse_cost( dnn_layer(input_layer(), w1_init, b1_init) );

  Vector x1(std::vector<double>({1.3,-0.3}));
  Vector y1(std::vector<double>({-0.5,1.7,-0.7}));
  
  Vector y1pred = w1_init * x1 + b1_init;
  double expect = pow(y1pred(0)-y1(0),2)/3. + pow(y1pred(1)-y1(1),2)/3. + pow(y1pred(2)-y1(2),2)/3.;
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
    
    data[i].x = Vector(1,x);
    data[i].y = Vector(1,0.2*x + 0.3);
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
  int ndata = 100;
  std::vector<XYpair> data(ndata);
  for(int i=0;i<ndata;i++){
    double eps = 2.0/(ndata - 1);
    double x = -1.0 + i*eps; //normalize x to within +-1
    
    data[i].x = Vector(1,x);
    data[i].y = Vector(1,0.2*x + 0.3);
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

      std::cout << "Test derivs x=" << data[d].x(0) << std::endl;
      for(int i=0;i<p.size(0);i++){
	Vector pp(p);
	pp(i) += 1e-9;
	model2.update(pp);
      
	double c2 = model2.loss(data[d].x,data[d].y);
	std::cout << i << " got " << pd(i) << " expect " << (c2-c1)/1e-9 << std::endl;
      }
    }
  }

#if 1
  DecayScheduler lr(0.001, 0.1);
  AdamParams ap;
  optimizeAdam(model, data, lr, ap, 200);

  std::cout << "Final params" << std::endl;
  Vector final_p = model.getParams();
  for(int i=0;i<final_p.size(0);i++)
    std::cout << i << " " << final_p(i) << std::endl;

  std::cout << "Test on some data" << std::endl;
  for(int d=0;d<data.size();d++){ //first 5 data    
    auto got = model.predict(data[d].x);
    std::cout << data[d].x(0) << " got " << got(0) << " expect " << data[d].y(0) << std::endl;
  }
#endif
}

int main(void){
  //basicTests();
  //testSimpleLinear();
  testOneHiddenLayer();

  return 0;
}

  
