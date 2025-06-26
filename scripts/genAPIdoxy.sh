#!/bin/bash
#Auto-generate docs/api_code.rst
#Execute from within docs directory
#pwd
for d in $(find ../include/ -type d); do
    cnt=$(ls $d/*.hpp 2>/dev/null | wc -l)
    if [ ${cnt} -gt 0 ]; then
	#echo $d $cnt
	../scripts/genDoxySpec.pl $d
    fi
done
#echo $dirs
