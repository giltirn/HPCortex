#pragma once
#include <Comms.hpp>

//Average an array over the ranks in the DDP communicator
template<typename FloatType>
void ddpAverage(FloatType* data, size_t len, bool pipeline_bcast = false);

#include "implementation/DDP.tcc"

#ifndef DDP_EXTERN_TEMPLATE_INST
#define SS extern
#else
#define SS
#endif
SS template void ddpAverage<float>(float* data, size_t len, bool pipeline_bcast);
SS template void ddpAverage<double>(double* data, size_t len, bool pipeline_bcast);
#undef SS
