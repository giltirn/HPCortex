SUBDIRS =
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_LDFLAGS = -Wl,-rpath=$(prefix)/lib
LDADD = -L$(top_builddir)/src -lhpcortex
testdir = $(prefix)/test/mpi
test_PROGRAMS = test_simple_linear_pipeline test_skip_connection_pipeline test_simple_linear_pipeline_ddp test_comms test_pipeline_1d test_conv1d_pipeline test_simple_linear_ddp 
test_simple_linear_pipeline_SOURCES = test_simple_linear_pipeline.cpp 
test_simple_linear_pipeline_LDADD = $(LDADD)
test_simple_linear_pipeline_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
test_skip_connection_pipeline_SOURCES = test_skip_connection_pipeline.cpp 
test_skip_connection_pipeline_LDADD = $(LDADD)
test_skip_connection_pipeline_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
test_simple_linear_pipeline_ddp_SOURCES = test_simple_linear_pipeline_ddp.cpp 
test_simple_linear_pipeline_ddp_LDADD = $(LDADD)
test_simple_linear_pipeline_ddp_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
test_comms_SOURCES = test_comms.cpp 
test_comms_LDADD = $(LDADD)
test_comms_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
test_pipeline_1d_SOURCES = test_pipeline_1d.cpp 
test_pipeline_1d_LDADD = $(LDADD)
test_pipeline_1d_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
test_conv1d_pipeline_SOURCES = test_conv1d_pipeline.cpp 
test_conv1d_pipeline_LDADD = $(LDADD)
test_conv1d_pipeline_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
test_simple_linear_ddp_SOURCES = test_simple_linear_ddp.cpp 
test_simple_linear_ddp_LDADD = $(LDADD)
test_simple_linear_ddp_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@

