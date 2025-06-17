#pragma once

#include <HPCortex.hpp>
#include <Testing.hpp>
#include <unordered_map>
#include <sstream>
#include <layers/TransformerEncoderDecoderBlock.hpp>

Vector<double> unitVec(int dim, int d_model){
  Vector<double> out(d_model, 0.);  
  autoView(out_v,out,HostReadWrite);
  out_v(dim) = 1.;
  return out;
}
std::vector<std::string> tokenize(const std::string &sentence){
  std::vector<std::string> tokens;
  std::istringstream iss(sentence);
  std::string token;
  while(iss >> token){
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string> buildContext(const std::vector<std::string> &sentence, const int C, bool shift_right = true){
  assert(sentence.size() <= C-1-(int)shift_right);
  std::vector<std::string> s(sentence);

  if(shift_right)
    s.insert(s.begin(), "<BOS>");
  
  s.push_back("<EOS>");
  while(s.size() < C)
    s.push_back("<PAD>");    
  
  assert(s.size() == C);
  return s;
}

Matrix<double> embed(const std::vector<std::string> &context,
		     const std::unordered_map<std::string, Vector<double> > &vocab_vec){
  int C = context.size();
  int d_model = vocab_vec.begin()->second.size(0);
  
  Matrix<double> out(C,d_model);
  for(int c=0;c<C;c++){
    auto it = vocab_vec.find(context[c]);
    if(it == vocab_vec.end()){
      std::cout << "Token: \"" << context[c] << "\" not in vocabulary" << std::endl;
      assert(0);
    }
    pokeRow(out, c, it->second);
  }
  return out;
}

std::string cat(const std::vector<std::string> &sentence){
  if(sentence.size()==0) return "";
  std::ostringstream os;
  for(int i=0;i<sentence.size()-1;i++)
    os << sentence[i] << " ";
  os << sentence.back();
  return os.str();
}

struct LogLossFunc{
  typedef std::vector< std::vector<std::string> const* > ComparisonType; //[batch_idx][token_idx]   data type input to the loss function for comparison
  typedef Tensor<double,3> PredictionType; //(token_idx, next_token_prob, batch_idx)    output of the model containing the prediction of the the next token for each input token

  std::unordered_map<std::string, int > const *token_idx_map;
  
  LogLossFunc(const std::unordered_map<std::string, int > &token_idx_map): token_idx_map(&token_idx_map){
  }

  double loss(const ComparisonType &y, const PredictionType &ypred) const{
    autoView(ypred_v,ypred,HostRead);

    int B = ypred.size(2);
    int P = ypred.size(1);
    int C = ypred.size(0);

    double ls = 0.;
    for(int b=0;b<B;b++){
      for(int c=0;c<C-1;c++){ //no "next" token for the last token; this token is always either EOS or PAD
	const std::string &next = (*y[b])[c+1];
	
	auto it = token_idx_map->find(next);
	assert(it != token_idx_map->end());
	int next_idx = it->second;
           
	std::cout << "c: " << c << " b: " << b << " next_tok: " << next << " idx: " << next_idx << " prob: " << ypred_v(c,next_idx,b) << std::endl;  
	
	ls -= log(ypred_v(c,next_idx,b)); //maximize the probability for the appropriate token
      }
    }
    return ls / B / (C-1);
  }

  //dCost/dypred(c,p,b)
  PredictionType layer_deriv(const ComparisonType &y, const PredictionType &ypred) const{
    PredictionType deriv(ypred.sizeArray(),0.,MemoryManager::Pool::HostPool);

    {
      autoView(ypred_v,ypred,HostRead);
      autoView(deriv_v,deriv,HostReadWrite);
      
      int B = ypred.size(2);
      int P = ypred.size(1);
      int C = ypred.size(0);

      for(int c=0;c<C-1;c++){
	for(int b=0;b<B;b++){
	  const std::string &next = (*y[b])[c+1];

	  auto it = token_idx_map->find(next);
	  assert(it != token_idx_map->end());
	  int next_idx = it->second;

	  deriv_v(c,next_idx,b) = -1./ypred_v(c,next_idx,b)/B/(C-1);	
	}
      }
    }
    return deriv;
  }
};
