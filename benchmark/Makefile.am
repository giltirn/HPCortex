SUBDIRS =
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_LDFLAGS = -Wl,-rpath=$(prefix)/lib
LDADD = -L$(top_builddir)/src -lmlcortex
benchmarkdir = $(prefix)/benchmark
benchmark_PROGRAMS = benchmark_dnn benchmark_activated_deriv benchmark_thin_mul_mat_mattranspose benchmark_mul_mattranspose_thinmat 
benchmark_dnn_SOURCES = benchmark_dnn.cpp 
benchmark_dnn_LDADD = $(LDADD)
benchmark_dnn_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
benchmark_activated_deriv_SOURCES = benchmark_activated_deriv.cpp 
benchmark_activated_deriv_LDADD = $(LDADD)
benchmark_activated_deriv_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
benchmark_thin_mul_mat_mattranspose_SOURCES = benchmark_thin_mul_mat_mattranspose.cpp 
benchmark_thin_mul_mat_mattranspose_LDADD = $(LDADD)
benchmark_thin_mul_mat_mattranspose_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
benchmark_mul_mattranspose_thinmat_SOURCES = benchmark_mul_mattranspose_thinmat.cpp 
benchmark_mul_mattranspose_thinmat_LDADD = $(LDADD)
benchmark_mul_mattranspose_thinmat_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@

