#Generate a Makefile for the benchmark directory (run from inside the directory)

out=Makefile.am
echo 'SUBDIRS =' > ${out}
echo 'AM_CPPFLAGS = -I$(top_srcdir)/include' >> ${out}
echo 'AM_LDFLAGS = -Wl,-rpath=$(prefix)/lib' >> ${out}
echo 'LDADD = -L$(top_builddir)/src -lhpcortex' >> ${out}
echo 'benchmarkdir = $(prefix)/benchmark' >> ${out}

echo -n 'benchmark_PROGRAMS = ' >> ${out}

for i in $(find . -name '*.cpp' | sed 's/^\.\///' | sed 's/\.cpp//'); do
    echo -n "$i " >> ${out}
done
echo '' >> ${out}

for i in $(find . -name '*.cpp' | sed 's/^\.\///' | sed 's/\.cpp//'); do
    echo "${i}_SOURCES = ${i}.cpp " >> ${out}
    echo "${i}_LDADD = \$(LDADD)" >> ${out}
    echo "${i}_LINK = \$(CXXLD) \$(AM_LDFLAGS) \$(LDFLAGS) -o \$@" >> ${out} #prevent autotools from including CXXFLAGS while linking!
done
echo '' >> ${out}
