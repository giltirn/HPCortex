#pragma once

//Average an array over the ranks in the DDP communicator
template<typename FloatType>
void ddpAverage(FloatType* data, size_t len, bool pipeline_bcast = false);

#include "implementation/DDP.tcc"
