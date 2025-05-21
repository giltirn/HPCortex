SUBDIRS =
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_LDFLAGS = -Wl,-rpath=$(prefix)/lib
LDADD = -L$(top_builddir)/src -lmlcortex
examplesdir = $(prefix)/examples
examples_PROGRAMS = example_train_embedding 
example_train_embedding_SOURCES = example_train_embedding.cpp 
example_train_embedding_LDADD = $(LDADD)
example_train_embedding_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@

