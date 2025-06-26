SUBDIRS =
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_LDFLAGS = -Wl,-rpath=$(prefix)/lib
LDADD = -L$(top_builddir)/src -lmlcortex
examplesdir = $(prefix)/examples
examples_PROGRAMS = example_transformer_encoder_decoder example_train_embedding example_dnn 
example_transformer_encoder_decoder_SOURCES = example_transformer_encoder_decoder.cpp 
example_transformer_encoder_decoder_LDADD = $(LDADD)
example_transformer_encoder_decoder_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
example_train_embedding_SOURCES = example_train_embedding.cpp 
example_train_embedding_LDADD = $(LDADD)
example_train_embedding_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@
example_dnn_SOURCES = example_dnn.cpp 
example_dnn_LDADD = $(LDADD)
example_dnn_LINK = $(CXXLD) $(AM_LDFLAGS) $(LDFLAGS) -o $@

