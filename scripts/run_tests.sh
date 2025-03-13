#!/bin/bash
set -e
rm -f run_tests.log
./test_accelerator 2>&1 | tee test_accelerator.log
./test_basic 2>&1 | tee test_basic.log
./test_activation 2>&1 | tee test_activation.log
mpirun -n  2 ./test_comms 2>&1 | tee test_comms.log
./test_managed_array 2>&1 | tee test_managed_array.log
./test_dynamic_model 2>&1 | tee test_dynamic_model.log
./test_memorymanager 2>&1 | tee test_memorymanager.log
./test_one_hidden_layer 2>&1 | tee test_one_hidden_layer.log
mpirun -n 2 ./test_pipeline_1d 2>&1 | tee test_pipeline_1d.log
./test_simple_linear 2>&1 | tee test_simple_linear.log
mpirun -n 2 ./test_simple_linear_ddp 2>&1 | tee test_simple_linear_ddp.log
mpirun -n 2 ./test_simple_linear_pipeline 2>&1 | tee test_simple_linear_pipeline.log
mpirun -n 4 ./test_simple_linear_pipeline_ddp 2>&1 | tee test_simple_linear_pipeline_ddp.log
