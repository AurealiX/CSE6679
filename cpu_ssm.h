#ifndef CPU_SSM_H
#define CPU_SSM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "transportation.h"

// CPU implementation of the Accelerated Stepping Stone Method for Phase 2 optimization.
// This function takes as input a TransportationProblem instance (with a valid initial BFS)
// and returns the elapsed time (in seconds) to reach optimality (or complete candidate evaluation).
double ssmCPUSolve(TransportationProblem *problem);

#ifdef __cplusplus
}
#endif

#endif
