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



AC_CONFIG_HEADERS([include/HPCortexConfig.h])

AC_CONFIG_SRCDIR([src/dummy])
AC_CONFIG_FILES([Makefile src/Makefile test/Makefile examples/Makefile benchmark/Makefile include/Makefile])

AC_OUTPUT
