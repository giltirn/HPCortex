#!/bin/bash
set -e
rm -f run_tests.log
./test_accelerator 2>&1 | tee test_accelerator.log
./test_basic 2>&1 | tee test_basic.log
./test_tensor 2>&1 | tee test_tensor.log
./test_optimizer 2>&1 | tee test_optimizer.log
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
./test_skip_connection 2>&1 | tee test_skip_connection.log
mpirun -n 2 ./test_skip_connection_pipeline 2>&1 | tee test_skip_connection_pipeline.log
./test_flatten_layer 2>&1 | tee test_flatten_layer.log
./test_conv1d 2>&1 | tee test_conv1d.log
./test_simple_linear_2d 2>&1 | tee test_simple_linear_2d.log
mpirun -n 3 ./test_conv1d_pipeline 2>&1 | tee test_conv1d_pipeline.log
./test_softmax 2>&1 | tee test_softmax.log
./test_matrix_tensor_contract 2>&1 | tee test_matrix_tensor_contract.log
./test_scaled_dotproduct_self_attention 2>&1 | tee test_scaled_dotproduct_self_attention.log
./test_multihead_self_attention 2>&1 | tee test_multihead_self_attention.log
./test_norm_layer 2>&1 | tee test_norm_layer.log
