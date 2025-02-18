#include<iostream>
#include<cmath>

//Experimentation with basic AST with forward differentiation
//sin(x1) + x1*x2
//sum( sin(x1), mult(x1,x2) )

//w1_1 = sin(x1)   //w1_2 = x1*x2
//Dw1_1/Dx1 = cos(x1)  Dw1_1/Dx2 = 0
//Dw1_2/Dx1 = x2  Dw1_2/Dx2 = x1

//w2 = sum(w1_1 , w1_2 )
//Dw2/Dw1_1 = 1   //Dw2/Dw1_2 = 1

//f=w2
//Df/Dw2 = 1

//Df/Dx1 = Df/Dw1 Dw1/Dx1
//       = Df/Dw2 ( Dw2/Dw1_1 Dw1_1/Dx1 + Dw2/Dw1_2 Dw1_2/Dx1 )
//       = 1 ( 1*cos(x1) + 1*x1 )

//Df/Dx2 = Df/Dw1 Dw1/Dx2
//       = Df/Dw2 ( Dw2/Dw1_1 Dw1_1/Dx2 + Dw2/Dw1_2 Dw1_2/Dx2 )
//       = 1 ( 1*0 + 1*x2 )



//f(x1,x2) = exp(cos(x1*x2) + log(x1-x2))


//w1_1 = sub(x1,x2)
//w1_2 = x1*x2

//w2_1 = cos(w1_2)
//w2_2 = log(w1_1)

//w3 = sum(w2_1, w2_2)

//w4 = exp(w3)

class OpBase{
public:
  virtual double value() const = 0;
  virtual double deriv(int varidx) const = 0;
  virtual ~OpBase(){}
}; 

class Input: public OpBase{
  int idx;
  double val;
public:
  Input(int idx, double val): idx(idx), val(val){}
  double value() const override{ return val; }
  double deriv(int varidx) const override{ return varidx == idx ? 1.0 : 0.0; }
};
class Scalar: public OpBase{
  double val;
public:
  Scalar(double val): val(val){}
  double value() const override{ return val; }
  double deriv(int varidx) const override{ return 0.; }
};


class Sin: public OpBase{
  double out;
  double dout;
  const OpBase &leaf;  
public:
  Sin(const OpBase &leaf): leaf(leaf){
    out = sin(leaf.value());
    dout = cos(leaf.value());
  }
  double value() const override{
    return out;
  }
  double deriv(int varidx) const  override{
    return dout * leaf.deriv(varidx);
  }
};

class Prod: public OpBase{
  double out;
  double dout1;
  double dout2;
  const OpBase &leaf1;
  const OpBase &leaf2;
public:
  Prod(const OpBase &leaf1, const OpBase &leaf2): leaf1(leaf1), leaf2(leaf2){
    out = leaf1.value()*leaf2.value();
    dout1 = leaf2.value();
    dout2 = leaf1.value();		     
  }
  double value() const override{
    return out;
  }
  double deriv(int varidx) const  override{
    return dout1 * leaf1.deriv(varidx) + dout2 * leaf2.deriv(varidx);
  }
};

class Sum: public OpBase{
  double out;
  const OpBase &leaf1;
  const OpBase &leaf2;
public:
  Sum(const OpBase &leaf1, const OpBase &leaf2): leaf1(leaf1), leaf2(leaf2){
    out = leaf1.value() + leaf2.value();
  }
  double value() const override{
    return out;
  }
  double deriv(int varidx) const  override{
    return leaf1.deriv(varidx) + leaf2.deriv(varidx);
  }
};

template<typename T>
struct OpBaseCon{
  T t;
};


struct test{
  const OpBase &a;

  test(const OpBase &a): a(a){
    std::cout << "ref" << std::endl;
  }
  test(OpBase &&a): a(std::move(a)){
    std::cout << "rref" << std::endl;
  }
  double value(){ return a.value(); }
};

template<typename T>
struct LeafStore{
  T v;
  LeafStore(T && v): v(v){
    std::cout << "STORE" << std::endl;
  }
};
template<typename T>
struct LeafRef{
  T &v;
  LeafRef(T &v): v(v){
    std::cout << "REF" << std::endl;
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




template<typename Store1, typename Store2>
class ProdT{
  double out;
  double dout1;
  double dout2;
  Store1 leaf1;
  Store2 leaf2;
public:
  ProdT(Store1 &&leaf1, Store2 &&leaf2): leaf1(leaf1), leaf2(leaf2){
    out = leaf1.v.value()*leaf2.v.value();
    dout1 = leaf2.v.value();
    dout2 = leaf1.v.value();		     
  }
  double value() const{
    return out;
  }
  double deriv(int varidx) const{
    return dout1 * leaf1.v.deriv(varidx) + dout2 * leaf2.v.deriv(varidx);
  }
};  

#define DDST(a) typename deduceStorage<decltype(a)>::type

template<typename U, typename V>
auto mult(U &&u, V &&v)->ProdT<DDST(u),DDST(v)>{
  return ProdT<DDST(u),DDST(v)>(std::forward<U>(u),std::forward<V>(v));
}
  


int main(void){
  Input ix1(0,0.1);
  Input ix2(1,0.2);
  Input ix3(2,0.3);

  auto p = mult(mult(ix1,ix2), ix3);

  std::cout << p.value() << std::endl;
  
  // const Input &l = ix1;
  // typename deduceStorage<decltype(l)>::type storel(l);
  // std::cout << storel.v.value() << std::endl;
  
  // typename deduceStorage<decltype(std::move(ix1))>::type storer(std::move(ix1));
  // std::cout << storer.v.value() << std::endl;

  // typename deduceStorage<decltype(ix1)>::type store3(ix1);
  // std::cout << store3.v.value() << std::endl;
  


  
  // Input ix1(0,0.1);
  // test t1(ix1);
  // std::cout << t1.value() << std::endl;
  
  // test t2(Input(0,0.1));
  // std::cout << t2.value() << std::endl;

  return 0;
}



#if 0
int main(void){
  double x1 = 1.31;
  double x2 = -4.55;
  
  //Forwards
  Input ix1(0,x1);
  Input ix2(1,x2);

  {
    Prod p(ix1,ix2);
    Sin s(ix1);
    
    Sum f(s,p);
    
    double expect = sin(x1) + x1*x2;
    double dexpect1 = cos(x1) + x2;
    double dexpect2 = x1;
    
    std::cout << f.value() << " " << expect << std::endl;
    std::cout << f.deriv(0) << " " << dexpect1 << std::endl;
    std::cout << f.deriv(1) << " " << dexpect2 << std::endl;
  }

  {
    //If the inputs are scalars and we are interested in the weights
    double a = 0.315;
    double b = -2.871; 
    Scalar ia(a);
    Scalar ib(b);

    Prod px1x2(ix1,ix2);
    Prod pbx1x2(ib, px1x2);
    Prod pax1(ia,ix1);
    Sin spax1(pax1);

    Sum f(spax1, pbx1x2);
    
    //Sum f( Sin(Prod(ia,ix1)) , Prod(ib,Prod(ix1,ix2)) );
    
    double expect = sin(a*x1) + b*x1*x2;
    double dexpect1 = cos(a*x1)*a + b*x2; //d/dx1
    double dexpect2 = b*x1;
    
    std::cout << f.value() << " " << expect << std::endl;
    std::cout << f.deriv(0) << " " << dexpect1 << std::endl;
    std::cout << f.deriv(1) << " " << dexpect2 << std::endl;
  }
  
  

  
  return 0;
}

#endif  
