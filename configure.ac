AC_INIT(HPCortex, 0.1)

AM_INIT_AUTOMAKE([-Wall foreign subdir-objects])
AC_LANG(C++)
AC_PROG_CXX
AM_PROG_AR
#LT_INIT

# Check for c++17
CXXFLAGS+=" -std=c++17"
AC_MSG_CHECKING([for c++17 compatibility])
AC_COMPILE_IFELSE( [AC_LANG_PROGRAM([[]],
                                    [[]])],
                   [AC_MSG_RESULT([yes])],
                   [AC_MSG_FAILURE(["Could not compile with CXXFLAGS $CXXFLAGS"])]
                  )

AC_CHECK_HEADER(mpi.h, [], [AC_MSG_FAILURE([["Cannot find MPI header"]])] )

AC_MSG_CHECKING([if CXX is NVCC])
AS_CASE([$CXX],
  [*nvcc*],
  [AC_MSG_RESULT([NVCC detected])
   using_nvcc=yes
   CXXFLAGS="$CXXFLAGS --extended-lambda"
   ],
  [AC_MSG_RESULT([NVCC not detected])
   using_nvcc=no
   ])

# Check for the --enable-cuda option
AC_ARG_ENABLE([cuda],
  [AS_HELP_STRING([--enable-cuda],
    [Enable CUDA support for device-side kernels])],
  [:],
  [enable_cuda=no])

if test "x$enable_cuda" = "xyes"; then
  AC_MSG_NOTICE([Enabling CUDA support])
  AC_DEFINE([USE_CUDA],[1],[Use CUDA])
fi

# We will use HIP_PLATFORM to detect if using the Nvidia backend, which changes how we pass compiler arguments
AC_ARG_VAR([HIP_PLATFORM], [HIP platform to target (e.g., nvidia, amd)])

# Check for the --enable-hip option
AC_ARG_ENABLE([hip],
  [AS_HELP_STRING([--enable-hip],
    [Enable HIP support for device-side kernels])],
  [:],
  [enable_hip=no])

if test "x$enable_hip" = "xyes"; then
  AC_MSG_NOTICE([Enabling HIP support])
  AC_DEFINE([USE_HIP],[1],[Use HIP])
fi

if test "x$enable_hip" = "xyes" && test "x$HIP_PLATFORM" = "xnvidia"; then
   CXXFLAGS="$CXXFLAGS --extended-lambda"
fi

# Check for the --enable-sycl option
AC_ARG_ENABLE([sycl],
  [AS_HELP_STRING([--enable-sycl],
    [Enable SYCL support for device-side kernels])],
  [:],
  [enable_sycl=no])

if test "x$enable_sycl" = "xyes"; then
  AC_MSG_NOTICE([Enabling SYCL support])
  AC_DEFINE([USE_SYCL],[1],[Use SYCL])
fi


# Check for the --enable-openmp option
AC_ARG_ENABLE([openmp],
  [AS_HELP_STRING([--enable-openmp],
    [Enable OpenMP support for host-side kernels])],
  [:],
  [enable_openmp=no])

if test "x$enable_openmp" = "xyes"; then
  AC_MSG_NOTICE([Enabling OpenMP support])
  if test "x$using_nvcc" = "xyes"; then
    CXXFLAGS="$CXXFLAGS -Xcompiler=-fopenmp"
    LDFLAGS="$LDFLAGS -Xcompiler=-fopenmp"
  elif test "x$enable_hip" = "xyes" && test "x$HIP_PLATFORM" = "xnvidia"; then
    CXXFLAGS="$CXXFLAGS -Xcompiler=-fopenmp"
    LDFLAGS="$LDFLAGS -Xcompiler=-fopenmp"
  else  
    CXXFLAGS="$CXXFLAGS -fopenmp"    
    LDFLAGS="$LDFLAGS -fopenmp"
  fi
  
  # Test if the compiler accepts OpenMP flags
  AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM(
      [[#include <omp.h>]],
      [[#pragma omp parallel for
        for (int i = 0; i < 10; ++i) {}]]
    )],
    [AC_MSG_RESULT([OpenMP support available])],
    [AC_MSG_RESULT([OpenMP support not available])
     AC_MSG_FAILURE([OpenMP support was requested, but your compiler does not appear to support it])])
fi

# Check for the --enable-cublas option
AC_ARG_ENABLE([cublas],
  [AS_HELP_STRING([--enable-cublas],
    [Enable CuBLAS support for high-performance matrix operations])],
  [:],
  [enable_cublas=no])

if test "x$enable_cublas" != "xno"; then
  AC_MSG_NOTICE([Enabling CuBLAS support])

  if test "x$enable_cuda" != "xyes"; then
    AC_MSG_FAILURE([--enable-cublas option requires compiling with CUDA support])
  fi

  if test "x$enable_cublas" != "xyes"; then
    #If not directly "yes", treat it as a path to the cuda installation
    CXXFLAGS="$CXXFLAGS -I$enable_cublas/include"
    LDFLAGS="$LDFLAGS -I$enable_cublas/lib64"
  fi

  LDFLAGS+=" -lcublas"

  AC_COMPILE_IFELSE(
    [
      AC_LANG_PROGRAM(
        [[#include<cublas_v2.h>]],
        [[cublasHandle_t handle; cublasCreate(&handle);]]
      )
    ],
    [
      AC_MSG_RESULT([CuBLAS support available])
      AC_DEFINE([USE_CUBLAS],[1],[Use CuBLAS])
      AC_DEFINE([USE_BLAS],[1],[Use BLAS])
    ],
    [AC_MSG_RESULT([CuBLAS support not available])
     AC_MSG_FAILURE([CuBLAS support was requested, but the test code did not compile])]
  )


fi


HPCORTEX_CXX="$CXX"
HPCORTEX_CXXLD="$CXXLD"
HPCORTEX_CXXFLAGS="$AM_CXXFLAGS $CXXFLAGS"
HPCORTEX_LDFLAGS="$AM_LDFLAGS $LDFLAGS"
HPCORTEX_LIBS=$LIBS

AC_SUBST([HPCORTEX_CXX])
AC_SUBST([HPCORTEX_CXXLD])
AC_SUBST([HPCORTEX_CXXFLAGS])
AC_SUBST([HPCORTEX_LDFLAGS])
AC_SUBST([HPCORTEX_LIBS])

AC_CONFIG_HEADERS([include/HPCortexConfig.h])

AC_CONFIG_SRCDIR([src/dummy])
AC_CONFIG_FILES([Makefile src/Makefile test/Makefile test/core/Makefile test/mpi/Makefile examples/Makefile benchmark/Makefile include/Makefile])
AC_CONFIG_FILES([hpcortex-config], [chmod +x hpcortex-config])

AC_OUTPUT
