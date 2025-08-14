#pragma once
#include <random>
#include <algorithm>
#include <limits>
#include <Tensors.hpp>
#include <Comms.hpp>
#include <DDP.hpp>
#include <Timing.hpp>

//LRscheduler must have the following method:
//FloatType operator()(const int epoch) const    :  return the learning rate for the given epoch

template<typename FloatType>
struct noScheduler{
  FloatType lr;
  noScheduler(FloatType lr): lr(lr){}
  
  FloatType operator()(const int epoch) const{ return lr; }
};

template<typename FloatType, typename LRscheduler = noScheduler<FloatType> >
class GradientDescentOptimizer{
  LRscheduler sched;
  FloatType eps;  
public:
  GradientDescentOptimizer(const LRscheduler &sched): sched(sched), eps(0.){}

  template<typename L=LRscheduler, typename std::enable_if<std::is_same<L,noScheduler<FloatType> >::value, int>::type = 0>
  GradientDescentOptimizer(FloatType lr): sched(lr){}  
  
  void epochStart(int epoch, bool verbose = true){
    eps = sched(epoch);
    if(verbose) std::cout << "GradientDescentOptimizer: Epoch " << epoch << " learning rate " << eps << std::endl;
  }
  inline Vector<FloatType> descentProfile(FloatType &step_size, const Vector<FloatType> &deriv) const{
    step_size = eps;
    return deriv;
  }

};

template<typename FloatType>
struct AdamParams{ //NB, alpha comes from the learning scheduler
  FloatType beta1;
  FloatType beta2;
  FloatType eps;
  AdamParams( FloatType beta1=0.99, FloatType beta2=0.999, FloatType eps=1e-8): beta1(beta1), beta2(beta2), eps(eps){}
};

template<typename FloatType, typename LRscheduler = noScheduler<FloatType> >
class AdamOptimizer{
  LRscheduler sched;
  AdamParams<FloatType> ap;
  FloatType alpha;
  size_t t;

  Vector<FloatType> m;
  Vector<FloatType> v;

  void reset(){
    t=0;
    m=Vector<FloatType>();
    v=Vector<FloatType>();
  }
public:
  AdamOptimizer(const AdamParams<FloatType> &ap, const LRscheduler &sched): sched(sched), ap(ap), alpha(0.), t(0){}
  AdamOptimizer(const LRscheduler &sched): AdamOptimizer( AdamParams<FloatType>(), sched){}
  
  template<typename L=LRscheduler, typename std::enable_if<std::is_same<L,noScheduler<FloatType> >::value, int>::type = 0>
  AdamOptimizer(const AdamParams<FloatType> &ap, FloatType lr): sched(lr), ap(ap), alpha(0.), t(0){}

  template<typename L=LRscheduler, typename std::enable_if<std::is_same<L,noScheduler<FloatType> >::value, int>::type = 0>
  AdamOptimizer(FloatType lr): AdamOptimizer( AdamParams<FloatType>(), lr){}
  
  void epochStart(int epoch, bool verbose = true){
    alpha = sched(epoch);
    if(epoch == 0) reset();
    if(verbose) std::cout << "AdamOptimizer: Epoch " << epoch << " learning rate (alpha) " << alpha << std::endl;
  }
  inline Vector<FloatType> descentProfile(FloatType &step_size, const Vector<FloatType> &g){
    int nparam = g.size(0);
    if(t==0){
      m = Vector<FloatType>(nparam,0);
      v = Vector<FloatType>(nparam,0);
    }
    Vector<FloatType> out(nparam);

    autoView(m_v,m,DeviceReadWrite);
    autoView(v_v,v,DeviceReadWrite);
    autoView(g_v,g,DeviceRead);
    autoView(out_v,out,DeviceWrite);

    auto b1 = ap.beta1;
    auto b2 = ap.beta2;
    auto eps = ap.eps;
    
    accelerator_for(p,nparam,{
      FloatType gp_init = g_v(p);
      m_v(p) = b1 * m_v(p) + (1.-b1)*g_v(p);
      v_v(p) = b2 * v_v(p) + (1.-b2)*pow(g_v(p),2);

      out_v(p) = m_v(p)/(sqrt(v_v(p)) + eps);
      });

    step_size =  t>0 ? alpha * sqrt(1. - pow(ap.beta2,t))  / (1. - pow(ap.beta1,t) ) : alpha;
    ++t;
    return out;
  }
};

template<typename FloatType>
class DecayScheduler{
  FloatType eps;
  FloatType decay_rate;
public:
  DecayScheduler(FloatType eps, FloatType decay_rate): eps(eps), decay_rate(decay_rate){}
  FloatType operator()(const int epoch) const{ return eps * 1./(1. + decay_rate * epoch); }
};

/**
 * @brief Train a model using DDP, whereby batches of data are distributed over ranks of the DDP communicator and trained in parallel
 * @param loss_func The model wrapper in a loss-function wrapper supporting calls to compute the loss and the loss derivative given a input/output data batch
 * @param data The training data loader, the spec for which is provided below
 * @param optimizer The optimizer
 * @param nepoch The number of epochs
 * @param batch_size The batch size
 * @param suppress_logging Optionally suppress logging output
 * @return The complete loss history for all batches / epochs
 *
 * DataLoaders are expected to contain the following methods:
 *    size_t size() const   :  return the total amount of data
 *    BatchType batch(int const* indices, int batch_size) const   : return the batch with batch size and indices as specified. BatchType must contain members 'x' and 'y', which are taken as the inputs to the model's loss function
 */
template<typename DataLoader, typename LossWrappedModelType, typename Optimizer>
std::vector<typename LossWrappedModelType::FloatType> train(LossWrappedModelType &loss_func, const DataLoader &data, Optimizer &optimizer, int nepoch, int batch_size, bool suppress_logging = false);

/**
 * @brief Train and validate model using DDP, whereby batches of data are distributed over ranks of the DDP communicator and trained in parallel
 * @param loss_func The model wrapper in a loss-function wrapper supporting calls to compute the loss and the loss derivative given a input/output data batch
 * @param train_data The training data loader (cf above for spec)
 * @param valid_data The validation data loader
 * @param optimizer The optimizer
 * @param nepoch The number of epochs
 * @param batch_size The batch size
 * @param suppress_logging Optionally suppress logging output
 * @return The complete loss history for all batches / epochs for training (first) and validation (second)
 */
template<typename DataLoader, typename LossWrappedModelType, typename Optimizer>
std::pair<std::vector<typename LossWrappedModelType::FloatType>, std::vector<typename LossWrappedModelType::FloatType> >
train(LossWrappedModelType &loss_func, const DataLoader &train_data, const DataLoader &valid_data, Optimizer &optimizer, int nepoch, int batch_size, bool suppress_logging = false);



template<typename FloatType, int DimX, int DimY>
struct XYpair{
  Tensor<FloatType,DimX> x;
  Tensor<FloatType,DimY> y;
};

//Insert data of with indices 'indices[i]' for i in 0..batch_size-1 into the last dimension of the output
template<typename FloatType, int DimX, int DimY>
inline XYpair<FloatType,DimX+1,DimY+1> batchData(int const* indices, int batch_size, const std::vector<XYpair<FloatType,DimX,DimY> > &data){
  assert(data.size()>0);
  int const *x_sz_in = data[0].x.sizeArray();
  int const *y_sz_in = data[0].y.sizeArray();
  
  int x_sz_out[DimX+1];
  int y_sz_out[DimY+1];
  memcpy(x_sz_out,x_sz_in,DimX*sizeof(int));
  memcpy(y_sz_out,y_sz_in,DimY*sizeof(int));
  x_sz_out[DimX] = batch_size;
  y_sz_out[DimY] = batch_size;
  
  XYpair<FloatType,DimX+1,DimY+1> out;
  out.x = Tensor<FloatType,DimX+1>(x_sz_out);
  out.y = Tensor<FloatType,DimY+1>(y_sz_out);

  for(int b=0;b<batch_size;b++){
    int i = indices[b];
    out.x.pokeLastDimension(data[i].x, b);
    out.y.pokeLastDimension(data[i].y, b);
  }
  return out;
}


template<typename FloatType, int DimX, int DimY>
class XYpairDataLoader{
  const std::vector<XYpair<FloatType,DimX,DimY> > &data;
public:
  XYpairDataLoader(const std::vector<XYpair<FloatType,DimX,DimY> > &data): data(data){}
  
  size_t size() const{ return data.size(); }
  
  XYpair<FloatType,DimX+1,DimY+1> batch(int const* indices, int batch_size) const{
    return batchData(indices,batch_size,data);
  }
};

#include "implementation/Optimizers.tcc"


