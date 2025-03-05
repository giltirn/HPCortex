#pragma once
#include <random>
#include <algorithm>
#include <Tensors.hpp>
#include <Comms.hpp>
#include <DDP.hpp>

template<typename FloatType, typename LRscheduler>
class GradientDescentOptimizer{
  const LRscheduler &sched;
  FloatType eps;  
public:
  GradientDescentOptimizer(const LRscheduler &sched): sched(sched), eps(0.){}
  
  void epochStart(int epoch){
    eps = sched(epoch);
    std::cout << "GradientDescentOptimizer: Epoch " << epoch << " learning rate " << eps << std::endl;
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

template<typename FloatType, typename LRscheduler>
class AdamOptimizer{
  const LRscheduler &sched;
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
  
  void epochStart(int epoch){
    alpha = sched(epoch);
    if(epoch == 0) reset();
    std::cout << "AdamOptimizer: Epoch " << epoch << " learning rate (alpha) " << alpha << std::endl;
  }
  inline Vector<FloatType> descentProfile(FloatType &step_size, const Vector<FloatType> &g){
    int nparam = g.size(0);
    if(t==0){
      m = Vector<FloatType>(nparam,0);
      v = Vector<FloatType>(nparam,0);
    }
    Vector<FloatType> out(nparam);

    autoView(m_v,m,HostReadWrite);
    autoView(v_v,v,HostReadWrite);
    autoView(g_v,g,HostRead);
    autoView(out_v,out,HostWrite);
    
    for(int p=0;p<nparam;p++){
      FloatType gp_init = g_v(p);
      m_v(p) = ap.beta1 * m_v(p) + (1.-ap.beta1)*g_v(p);
      v_v(p) = ap.beta2 * v_v(p) + (1.-ap.beta2)*pow(g_v(p),2);

      out_v(p) = m_v(p)/(sqrt(v_v(p)) + ap.eps);
    }

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

template<typename FloatType>
struct XYpair{
  Vector<FloatType> x;
  Vector<FloatType> y;
};
template<typename FloatType>
struct batchedXYpair{
  Matrix<FloatType> x;
  Matrix<FloatType> y;
};

template<typename FloatType>
inline batchedXYpair<FloatType> batchData(int* indices, int batch_size, const std::vector<XYpair<FloatType> > &data){
  assert(data.size()>0);
  int x_features = data[0].x.size(0);
  int y_features = data[0].y.size(0);
  
  batchedXYpair<FloatType> out;
  out.x = Matrix<FloatType>(x_features, batch_size);
  out.y = Matrix<FloatType>(y_features, batch_size);

  for(int b=0;b<batch_size;b++){
    int i = indices[b];
    out.x.pokeColumn(b, data[i].x);
    out.y.pokeColumn(b, data[i].y);
  }
  return out;
}


template<typename FloatType, typename ModelType, typename Optimizer>
std::vector<FloatType> train(ModelType &model, const std::vector<XYpair<FloatType> > &data, Optimizer &optimizer, int nepoch, int batch_size){
  std::default_random_engine gen(1234); //important that every rank shuffles in the same way

  //We want to divide the data evenly over batches. This means we may need to discard some data
  int nbatch = data.size() / batch_size;
  int ndata = nbatch * batch_size;
  std::uniform_int_distribution<int> dist(0,ndata-1);
  
  std::vector<int> didx(ndata);
  for(int i=0;i<ndata;i++) didx[i] = i;

  int nparam = model.nparams();

  //For DDP we solve blocks of batches in parallel
  int blocksize = communicators().ddpNrank();
  int nblocks = (nbatch + blocksize - 1) / blocksize; //round up
  int me = communicators().ddpRank(); //all ranks in a pipeline will have the same value for the ddp rank, but only the pipeline leader should communicate

  bool do_print = me == 0 && communicators().isPipelineLeader();

  if(do_print) std::cout << "Training with " << ndata << " data samples divided into " << nbatch << " batches of size " << batch_size
			 << " using DDP over " << blocksize << " ranks with " << nblocks << " iterations per epoch" << std::endl;
  
  std::vector<FloatType> losses(nblocks*nepoch);
    
  for(int epoch=0;epoch<nepoch;epoch++){
    optimizer.epochStart(epoch);
    std::random_shuffle ( didx.begin(), didx.end(), [&](const int l){ return dist(gen); }  );

    for(int block=0;block<nblocks;block++){
      int blocksize_actual = std::min(nbatch - block*blocksize, blocksize);

      FloatType loss = 0;
      Vector<FloatType> deriv(nparam, 0.);
      
      if(me < blocksize_actual){ //if not enough data to have all ranks do work in this block
	int bidx = block*blocksize + me; //which batch are we doing?

	//Get the batch
	batchedXYpair<FloatType> bxy = batchData(didx.data() + bidx*batch_size, batch_size, data);
	
	loss = model.loss(bxy.x, bxy.y);
	deriv = model.deriv();
      }
      ddpAverage(&loss,1,false); //no need to bcast the loss to the pipeline ranks
      ddpAverage(deriv,true); //share the deriv over all pipeline ranks
           
      if(do_print) std::cout << epoch << "-" << block << " : "<< loss << std::endl;
      
      FloatType eps;
      Vector<FloatType> direction = optimizer.descentProfile(eps,deriv);
      
      model.step( direction, eps );

      losses[block+nblocks*epoch] = loss;
    }
  }
  return losses;
}


