AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include

# Enable position-independent code
AM_CXXFLAGS = -fPIC

# List your C++ source files here.
sources = Init.cpp DDP.cpp Tensors.cpp MemoryManager.cpp Accelerator.cpp LossFunctions.cpp Comms.cpp ActivationFuncs.cpp
# Convert .cpp sources to object files.
objects = $(sources:.cpp=.o)

# Name of the shared library.
libname = libmlcortex.so

# Default target.
all: $(libname)

%.o : %.cpp
	$(CXX) $(AM_CXXFLAGS) $(AM_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Link the shared library.
$(libname): $(objects)
	$(CXX) -shared -o $(libname) $(objects) $(LDFLAGS)

# Installation: install the shared library to $(prefix)/lib.
libdir = $(prefix)/lib

install-exec-hook:
	$(INSTALL) -d $(DESTDIR)$(libdir)
	$(INSTALL) -m 755 $(libname) $(DESTDIR)$(libdir)

uninstall-hook:
	rm -f $(DESTDIR)$(libdir)/$(libname)

# Clean up generated files.
clean-local:
	rm -f $(objects) $(libname)

