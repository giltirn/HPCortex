## Dependencies

  * MPI
  * OpenMP
  * CUDA (Optional)

The non-GPU implementation is primarily for testing and development, and is not likely to achieve optimal performance

## Installation

CUDA + OpenMP (for NVIDIA V100 with compute capability 7.0):   

	./autogen.sh
    CXXFLAGS="--forward-unknown-to-host-compiler -x cu -ccbin mpic++ -gencode=arch=compute_70,code=sm_70 -g" \
    LDFLAGS="--forward-unknown-to-host-compiler -link -ccbin mpic++ -gencode=arch=compute_70,code=sm_70 -g -ldl" \
    CXX="nvcc" \
    ./configure --prefix=/path/to/install/dir --enable-openmp --enable-cuda`
    make -j6
    make install
	
Pure OpenMP:  

	./autogen.sh
    CXXFLAGS=" -g" \
    LDFLAGS="-g" \
    CXX="mpic++" \
    ./configure --prefix=/path/to/install/dir --enable-openmp
    make -j6
    make install
