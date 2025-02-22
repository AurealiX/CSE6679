#ifndef CPU_MODI_H
#define CPU_MODI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "transportation.h"

// Solve the Phase 2 optimization by MODI method.
// Return the running time
double modiCPUSolve(TransportationProblem* problem);

#ifdef __cplusplus
}
#endif

#endif 
