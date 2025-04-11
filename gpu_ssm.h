#ifndef GPU_SSM_H
#define GPU_SSM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "transportation.h"  // Provides the TransportationProblem structure

// GPU-accelerated Stepping Stone Method (SSM) for phase 2 optimization.
// This function implements the BFS-based search, backtrace, and reduction steps
// for identifying a candidate with negative reduced cost, and eventually pivoting.
// It returns the elapsed time in seconds.
double ssmGPUSolve(TransportationProblem* problem);

#ifdef __cplusplus
}
#endif

#endif // GPU_SSM_H
