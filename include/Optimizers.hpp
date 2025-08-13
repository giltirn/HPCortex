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

//DataLoaders are expected to contain the following methods:
//size_t size() const   :  return the total amount of data
//BatchType batch(int const* indices, int batch_size) const   : return the batch with batch size and indices as specified. BatchType must contain members 'x' and 'y', which are taken as the inputs to the model's loss function
template<typename DataLoader, typename LossWrappedModelType, typename Optimizer>
std::vector<typename LossWrappedModelType::FloatType> train(LossWrappedModelType &loss_func, const DataLoader &data, Optimizer &optimizer, int nepoch, int batch_size, bool suppress_logging = false){
  typedef typename LossWrappedModelType::FloatType FloatType;
  std::default_random_engine gen(1234); //important that every rank shuffles in the same way

  //We want to divide the data evenly over batches. This means we may need to discard some data
  if(batch_size > data.size())
    batch_size = data.size();

  int nbatch = data.size() / batch_size;
  int ndata = nbatch * batch_size;
  std::uniform_int_distribution<int> dist(0,ndata-1);
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;

  int nparam = loss_func.nparams();

  //For DDP we solve blocks of batches in parallel
  int blocksize = communicators().ddpNrank();
  int nblocks = (nbatch + blocksize - 1) / blocksize; //round up
  int me = communicators().ddpRank(); //all ranks in a pipeline will have the same value for the ddp rank, but only the pipeline leader should communicate

  bool do_print = me == 0 && communicators().isPipelineLeader() && !suppress_logging ;

  if(do_print) std::cout << "Training with " << ndata << " data samples divided into " << nbatch << " batches of size " << batch_size
			 << " using DDP over " << blocksize << " ranks with " << nblocks << " iterations per epoch" << std::endl;
  
  std::vector<FloatType> losses(nblocks*nepoch);
    
  for(int epoch=0;epoch<nepoch;epoch++){
    optimizer.epochStart(epoch, do_print);
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );

    FloatType lmax=std::numeric_limits<FloatType>::lowest(),  lmin = std::numeric_limits<FloatType>::max(),  lavg = 0.;
    auto ts=now();
    
    for(int block=0;block<nblocks;block++){
      int blocksize_actual = std::min(nbatch - block*blocksize, blocksize);

      FloatType loss = 0;
      Vector<FloatType> deriv(nparam, 0.);
      
      if(me < blocksize_actual){ //if not enough data to have all ranks do work in this block
	int bidx = block*blocksize + me; //which batch are we doing?

	//Get the batch
	auto bxy = data.batch(didx.data() + bidx*batch_size, batch_size);

	loss = loss_func.loss(bxy.x, bxy.y, DerivYes);
	deriv = loss_func.deriv();
      }
      ddpAverage(&loss,1,false); //no need to bcast the loss to the pipeline ranks
      ddpAverage(deriv,true); //share the deriv over all pipeline ranks
           
      //if(do_print) std::cout << epoch << "-" << block << " : "<< loss << std::endl;
      lmax = std::max(lmax,loss);
      lmin = std::min(lmin,loss);
      lavg += loss;
      
      FloatType eps;
      Vector<FloatType> direction = optimizer.descentProfile(eps,deriv);
      
      loss_func.step( direction, eps );

      losses[block+nblocks*epoch] = loss;
    }
    lavg /= nblocks;
    if(do_print) std::cout << "Epoch : " << epoch << " time : " << since(ts) <<"s " << " loss min: " << lmin << " avg: " << lavg << " max: " << lmax << std::endl;    
  }
  return losses;
}


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

template<typename FloatType, int DimX, int DimY, typename LossWrappedModelType, typename Optimizer>
std::vector<FloatType> train(LossWrappedModelType &loss_func, const std::vector<XYpair<FloatType,DimX,DimY> > &data, Optimizer &optimizer, int nepoch, int batch_size, bool suppress_logging = false){
  XYpairDataLoader<FloatType,DimX,DimY> loader(data);
  return train(loss_func, loader, optimizer, nepoch, batch_size, suppress_logging);
}




