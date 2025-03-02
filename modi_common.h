// modi_common.h
#ifndef MODI_COMMON_H
#define MODI_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include "transportation.h"

// Function declarations (without 'static')
int inPath(int path[][2], int path_len, int r, int c);
int dfs_find_loop(TransportationProblem *problem, int m, int n,
                  int start_row, int start_col,
                  int curr_row, int curr_col,
                  int move_dir, int depth,
                  int path[][2], int *loop_length);
int findLoop(TransportationProblem *problem, int m, int n,
             int start_row, int start_col, int loop[][2], int *loop_length);
void epsilonAllocation(TransportationProblem* problem, double epsilon);
void computePotentials(TransportationProblem *problem, double *u, double *v, int m, int n);

#ifdef __cplusplus
}
#endif

#endif
