out=Makefile.am

cat <<'EOF' > ${out}
AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include

# Enable position-independent code
AM_CXXFLAGS = -fPIC

# List your C++ source files here.
EOF

echo -n "sources =" >> ${out}
for i in $(find . -name '*.cpp' | sed 's/^\.\///'); do
     echo -n " $i" >> ${out}
done

cat <<'EOF' >> ${out}

# Convert .cpp sources to object files.
objects = $(sources:.cpp=.o)

# Name of the shared library.
libname = libhpcortex.so

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

EOF

