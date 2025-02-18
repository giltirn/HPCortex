#include<iostream>
#include<cmath>

//Similar to diff.cc but use expression templates rather than inheritance for cleaner interface
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

struct LeafTag{};

class Input{
  int idx;
  double val;
public:
  typedef LeafTag tag;
  
  Input(int idx, double val): idx(idx), val(val){}
  double value() const{ return val; }
  double deriv(int varidx) const{ return varidx == idx ? 1.0 : 0.0; }
};
class Scalar{
  double val;
public:
  typedef LeafTag tag;
  
  Scalar(double val): val(val){}
  double value() const { return val; }
  double deriv(int varidx) const { return 0.; }
};

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

template<typename Store1, typename Store2>
class ProdT{
  double out;
  double dout1;
  double dout2;
  Store1 leaf1;
  Store2 leaf2;
public:
  typedef LeafTag tag;
  
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

template<typename U, typename V, typename std::enable_if<ISLEAF(U) && ISLEAF(V), int>::type = 0>
auto operator*(U &&u, V &&v)->ProdT<DDST(u),DDST(v)>{
  return ProdT<DDST(u),DDST(v)>(std::forward<U>(u),std::forward<V>(v));
}

template<typename Store1, typename Store2>
class SumT{
  double out;
  Store1 leaf1;
  Store2 leaf2;
public:
  typedef LeafTag tag;
  
  SumT(Store1 &&leaf1, Store2 &&leaf2): leaf1(leaf1), leaf2(leaf2){
    out = leaf1.v.value() + leaf2.v.value();
  }
  double value() const{
    return out;
  }
  double deriv(int varidx) const{
    return leaf1.v.deriv(varidx) + leaf2.v.deriv(varidx);
  }
};  

template<typename U, typename V, typename std::enable_if<ISLEAF(U) && ISLEAF(V), int>::type = 0>
auto operator+(U &&u, V &&v)->SumT<DDST(u),DDST(v)>{
  return SumT<DDST(u),DDST(v)>(std::forward<U>(u),std::forward<V>(v));
}


template<typename Store>
class SinT{
  double out;
  double dout;
  Store leaf;  
public:
  typedef LeafTag tag;
  
  SinT(Store &&leaf): leaf(leaf){
    out = sin(leaf.v.value());
    dout = cos(leaf.v.value());
  }
  double value() const{
    return out;
  }
  double deriv(int varidx) const{
    return dout * leaf.v.deriv(varidx);
  }
};

template<typename U, typename std::enable_if<ISLEAF(U), int>::type = 0>
auto sin(U &&u)->SinT<DDST(u)>{
  return SinT<DDST(u)>(std::forward<U>(u));
}


// struct Test1{
//   typedef LeafTag tag;
// };

// template<typename T, typename U, typename std::enable_if<std::is_same<typename std::decay<T>::type::tag,LeafTag>::value && std::is_same<typename std::decay<U>::type::tag,LeafTag>::value, int>::type = 0>
// T operator*(T &&t, U &&u){}


int main(void){ 
  double x1 = 1.31;
  double x2 = -4.55;

  Input ix1(0,x1);
  Input ix2(1,x2);

  {
    //auto f = sum( Sin(ix1) , ix1*ix2 );
    auto f = sin(ix1) + ix1*ix2;
  
    double expect = sin(x1) + x1*x2;
    double dexpect1 = cos(x1) + x2;
    double dexpect2 = x1;
    
    std::cout << f.value() << " " << expect << std::endl;
    std::cout << f.deriv(0) << " " << dexpect1 << std::endl;
    std::cout << f.deriv(1) << " " << dexpect2 << std::endl;
  }

  // {
  //   //If the inputs are scalars and we are interested in the weights
  //   double a = 0.315;
  //   double b = -2.871; 
  //   Scalar ia(a);
  //   Scalar ib(b);

  //   Prod px1x2(ix1,ix2);
  //   Prod pbx1x2(ib, px1x2);
  //   Prod pax1(ia,ix1);
  //   Sin spax1(pax1);

  //   Sum f(spax1, pbx1x2);
    
  //   //Sum f( Sin(Prod(ia,ix1)) , Prod(ib,Prod(ix1,ix2)) );
    
  //   double expect = sin(a*x1) + b*x1*x2;
  //   double dexpect1 = cos(a*x1)*a + b*x2; //d/dx1
  //   double dexpect2 = b*x1;
    
  //   std::cout << f.value() << " " << expect << std::endl;
  //   std::cout << f.deriv(0) << " " << dexpect1 << std::endl;
  //   std::cout << f.deriv(1) << " " << dexpect2 << std::endl;
  // }
  
  

  
  return 0;
}

