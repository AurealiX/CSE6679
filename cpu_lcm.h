#ifndef CPU_LCM_H
#define CPU_LCM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "transportation.h"

// CPU implementation of the Least Cost Method for Phase 1.
// Returns the elapsed time (in seconds) for computing the initial BFS.
double lcmCPUSolve(TransportationProblem *problem);

#ifdef __cplusplus
}
#endif

#endif
