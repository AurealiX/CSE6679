// gpu_modi.h
#ifndef GPU_MODI_H
#define GPU_MODI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "transportation.h"

// Declaration for the GPU-accelerated MODI solver
double modiGPUSolve(TransportationProblem* problem);

#ifdef __cplusplus
}
#endif

#endif