#ifndef GPU_VAM_H
#define GPU_VAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "transportation.h"

// GPU VAM solver: returns elapsed time in seconds.
double gpuVamSolve(TransportationProblem* problem);

#ifdef __cplusplus
}
#endif

#endif
