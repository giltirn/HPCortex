#!/bin/bash
set -e

if [[ $# == 0 || ( $# == 1 && $1 == "core" ) ]]; then
    echo "Running core tests"
    cd core
    for i in $(find . -maxdepth 1 -type f -executable); do
	./${i} 2>&1 | tee "${i}.log"
    done
    cd -

fi

if [[ $# == 0 || ( $# == 1 && $1 == "mpi" ) ]]; then
    echo "Running mpi tests"
    cd mpi
    mpirun -n  2 ./test_comms 2>&1 | tee test_comms.log
    mpirun -n 2 ./test_pipeline_1d 2>&1 | tee test_pipeline_1d.log
    mpirun -n 2 ./test_simple_linear_ddp 2>&1 | tee test_simple_linear_ddp.log
    mpirun -n 2 ./test_simple_linear_pipeline 2>&1 | tee test_simple_linear_pipeline.log
    mpirun -n 4 ./test_simple_linear_pipeline_ddp 2>&1 | tee test_simple_linear_pipeline_ddp.log
    mpirun -n 2 ./test_skip_connection_pipeline 2>&1 | tee test_skip_connection_pipeline.log
    mpirun -n 3 ./test_conv1d_pipeline 2>&1 | tee test_conv1d_pipeline.log
    cd -
    
fi
