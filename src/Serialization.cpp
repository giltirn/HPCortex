#include <Serialization.hpp>

Endianness endianness(){
  static int e = -1;
  if(e==-1){
    union {
      int i;
      uint8_t c[sizeof(int)];
    } u;
    u.i = 1;
    if (u.c[0] == 1)
      e = 0; //little endian
    else
      e = 1;
  }
  return e == 0 ? Endianness::Little : Endianness::Big;
}


//Courtesy of https://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable
static const uint8_t _BitReverseTable256[256] = 
{
#   define R2(n)     n,     n + 2*64,     n + 1*64,     n + 3*64
#   define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#   define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
    R6(0), R6(2), R6(1), R6(3)
#undef R2
#undef R4
#undef R6   
};

uint8_t BitReverseTable256(size_t i){
  return _BitReverseTable256[i];
}

std::string toString(const Endianness e){
  switch(e){
  case Endianness::Big:
    return "Big";
  case Endianness::Little:
    return "Little";
  case Endianness::System:
    return "System";
  }
  return "";
}
 
BinaryWriter::BinaryWriter(const std::string &filename, const Endianness end): do_flip(false){
  if(end != Endianness::System && end != endianness()) do_flip = true;
  std::cout << "Writing to " << filename << " in " << toString( end == Endianness::System ? endianness() : end) << " endian" << std::endl;
  of.open(filename, std::ios::binary);
  writeValue((uint8_t)1); //allow inference of endianness
  writeValue(3.14159f); //allow test of inference!    
}


BinaryReader::BinaryReader(const std::string &filename): do_flip(false){      
  of.open(filename, std::ios::binary);
  uint8_t e;
  of.read((char*)&e, 1); assert(of.good());
  if(e == 1) do_flip = false; //same endianness as system
  else if(e == 128) do_flip = true;
  else{
    std::cout << "Read e " << e << std::endl;
    assert(0);
  }
  std::cout << "Input file endianness" << (do_flip ? " different from " : " same as ") << "system's" << std::endl;
  Endianness end_in = endianness();    
  if(do_flip && endianness() == Endianness::Little) end_in = Endianness::Big;
  else if(do_flip && endianness() == Endianness::Big) end_in = Endianness::Little;
    
  std::cout << "Reading from " << filename << " in " << toString(end_in) << " endian" << std::endl;
    
  float ck = readValue<float>();
  std::cout << "Read ck " << ck << " " << ck - 3.14159f << std::endl;
  assert(ck == 3.14159f);
}
