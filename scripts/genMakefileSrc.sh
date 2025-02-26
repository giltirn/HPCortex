out=Makefile.am

echo 'AM_CPPFLAGS = -I$(top_srcdir)/include' > ${out}
echo 'lib_LTLIBRARIES = libmlcortex.la' >> ${out}
echo -n 'libmlcortex_la_SOURCES =' >> ${out}

for i in $(find . -name '*.cpp' | sed 's/^\.\///'); do
    echo -n " $i" >> ${out}
done
echo  "" >> ${out}

echo 'libmlcortex_la_LDFLAGS = -version-info 0:0:0' >> ${out}
