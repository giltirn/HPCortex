## Dependencies

  * MPI
  * OpenMP
  * CUDA, HIP, SYCL (Optional)
  * cuBLAS (CUDA), rocBLAS (HIP), oneMKL (SYCL)  (Optional)

The non-GPU implementation is primarily for testing and development, and is not likely to achieve optimal performance

## Installation

# CUDA + OpenMP (for NVIDIA V100 with compute capability 7.0):   

	./autogen.sh
    CXXFLAGS="--forward-unknown-to-host-compiler -x cu -ccbin mpic++ -gencode=arch=compute_70,code=sm_70 -g" \
    LDFLAGS="--forward-unknown-to-host-compiler -link -ccbin mpic++ -gencode=arch=compute_70,code=sm_70 -g -ldl" \
    CXX="nvcc" \
    ./configure --prefix=/path/to/install/dir --enable-openmp --enable-cuda
    make -j6
    make install

Optionally enable cuBLAS with `--enable-cublas`

# Pure OpenMP:  

	./autogen.sh
    CXXFLAGS=" -g" \
    LDFLAGS="-g" \
    CXX="mpic++" \
    ./configure --prefix=/path/to/install/dir --enable-openmp
    make -j6
    make install

# OLCF Frontier:

	./autogen.sh
	module load PrgEnv-amd rocm craype-accel-amd-gfx90a
    CXX=CC \
    CXXFLAGS="-x hip -D__HIP_ARCH_GFX90A__=1 --offload-arch=gfx90a -O3" \
    ./configure --prefix=/path/to/install/dir --enable-hip --enable-openmp --enable-rocblas
    make -j 8
    make install

# ALCF Aurora:

	./autogen.sh
    CXX=mpic++ \
    CXXFLAGS="-O3 -g -fsycl -fsycl-targets=spir64" \
    LDFLAGS="-O3 -g -fsycl -fsycl-targets=spir64" \
    ./configure --prefix=/path/to/install/dir --enable-sycl --enable-onemkl


