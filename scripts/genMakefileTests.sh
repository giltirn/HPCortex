#SUBDIRS =

#AM_CPPFLAGS = -I$(top_srcdir)/include
#LDADD = ../src/libmlcortex.la

#testdir = $(prefix)/test
#test_PROGRAMS = test_pipeline_1d

#test_pipeline_1d_SOURCES = test_pipeline_1d.cpp
#test_pipeline_1d_LDADD = $(LDADD)




#Generate a Makefile for the src directory (run from inside the directory)

out=Makefile.am
echo 'SUBDIRS =' > ${out}
echo 'AM_CPPFLAGS = -I$(top_srcdir)/include' >> ${out}
echo 'LDADD = ../src/libmlcortex.la' >> ${out}
echo 'testdir = $(prefix)/test' >> ${out}

echo -n 'test_PROGRAMS = ' >> ${out}

for i in $(find . -name '*.cpp' | sed 's/^\.\///' | sed 's/\.cpp//'); do
    echo -n "$i " >> ${out}
done
echo '' >> ${out}

for i in $(find . -name '*.cpp' | sed 's/^\.\///' | sed 's/\.cpp//'); do
    echo "${i}_SOURCES = ${i}.cpp " >> ${out}
    echo "${i}_LDADD = \$(LDADD)" >> ${out}
done
echo '' >> ${out}



#echo 'AM_CPPFLAGS = -I$(top_srcdir)/include -I$(top_srcdir)/3rdparty @PS_FLAGS@' > ${out}
#echo 'lib_LTLIBRARIES = libchimbuko.la' >> ${out}
#echo -n 'libchimbuko_la_SOURCES = ' >> ${out}

#for i in $(find . -name '*.cpp' | sed 's/^\.\///'); do
#    echo -n "$i " >> ${out}
#done
#echo '' >> ${out}

#echo 'libchimbuko_la_LDFLAGS = -version-info 3:0:0' >> ${out}
