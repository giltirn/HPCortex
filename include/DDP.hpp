#pragma once

//Average an array over the ranks in the DDP communicator
void ddpAverage(double* data, size_t len, bool pipeline_bcast = false);
