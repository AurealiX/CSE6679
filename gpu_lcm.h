#ifndef GPU_LCM_H
#define GPU_LCM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "transportation.h"  // Provides the definition of TransportationProblem

// GPU implementation of the Least Cost Method (LCM) for Phase 1 of the Transportation Problem.
// Returns the elapsed time in seconds.
double lcmGPUSolve(TransportationProblem *problem);

#ifdef __cplusplus
}
#endif

#endif // GPU_LCM_H
