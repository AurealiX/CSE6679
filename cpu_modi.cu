#include "cpu_modi.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>


// Helper function to check if a cell (r, c) is already in the path.
static int inPath(int path[][2], int path_len, int r, int c) {
    for (int i = 0; i < path_len; i++) {
        if (path[i][0] == r && path[i][1] == c)
            return 1;
    }
    return 0;
}

// Recursive DFS to find a closed loop (cycle) starting from (start_row, start_col).
// move_dir: 0 = horizontal move next; 1 = vertical move next.
// When a loop is found (with at least 4 cells, even in length, returning to the start),
// the loop length is stored in loop_length.
static int dfs_find_loop(TransportationProblem *problem, int m, int n,
                         int start_row, int start_col,
                         int curr_row, int curr_col,
                         int move_dir, int depth,
                         int path[][2], int *loop_length) {
    // printf("current loc: (%d,%d); depth %d\n",curr_row,curr_col,depth);
    if (depth >= 4 && curr_row == start_row && curr_col == start_col && (depth % 2 == 0)) {
        *loop_length = depth;
        return 1;
    }
    if (depth >= m+n)
        return 0;

    if (move_dir == 0) { // horizontal move: vary column.
        for (int col = 0; col < n; col++) {
            if (col == curr_col)
                continue;
            // Valid if the cell is basic (allocated) or it is the candidate cell.
            if (!((problem->BFS[curr_row][col]==1) ||
                  (curr_row == start_row && col == start_col)))
                continue;
            if (inPath(path, depth, curr_row, col) && !(curr_row == start_row && col == start_col))
                continue;
            path[depth+1][0] = curr_row;
            path[depth+1][1] = col;
            if (dfs_find_loop(problem, m, n, start_row, start_col,
                              curr_row, col, 1, depth + 1, path, loop_length))
                return 1;
        }
    } else { // vertical move: vary row.
        for (int row = 0; row < m; row++) {
            if (row == curr_row)
                continue;
            if (!((problem->BFS[row][curr_col]==1) ||
                  (row == start_row && curr_col == start_col)))
                continue;
            if (inPath(path, depth, row, curr_col) && !(row == start_row && curr_col == start_col))
                continue;
            path[depth+1][0] = row;
            path[depth+1][1] = curr_col;
            if (dfs_find_loop(problem, m, n, start_row, start_col,
                              row, curr_col, 0, depth + 1, path, loop_length))
                return 1;
        }
    }
    return 0;
}

// Attempts to find a closed loop for the candidate cell (start_row, start_col).
// On success, copies the loop into 'loop' and sets loop_length.
static int findLoop(TransportationProblem *problem, int m, int n,
                    int start_row, int start_col, int loop[][2], int *loop_length) {
    int path[m+n][2];
    path[0][0] = start_row;
    path[0][1] = start_col;
    // int movedir=random()%2;
    int movedir=0;
    if (dfs_find_loop(problem, m, n, start_row, start_col, start_row, start_col, movedir, 0, path, loop_length)) {
        memcpy(loop, path, (*loop_length) * 2 * sizeof(int));
        return 1;
    }
    return 0;
}

// Prevent degeneracy
void epsilonAllocation(TransportationProblem* problem, double epsilon){
    for(int i=0;i<problem->numSupply;i++){
        for(int j=0;j<problem->numDemand;j++){
            if(problem->BFS[i][j]==1&&problem->solution[i][j]<epsilon)
                problem->solution[i][j]+=epsilon;
        }
    }
}

// Compute the dual potentials u (for rows) and v (for columns) for the current basic solution.
// Basic cells are those with an allocation greater than 1e-5.
static void computePotentials(TransportationProblem *problem, double *u, double *v, int m, int n) {
    // Initialize potentials with a sentinel value.
    for (int i = 0; i < m; i++) {
        u[i] = FLT_MAX;
    }
    for (int j = 0; j < n; j++) {
        v[j] = FLT_MAX;
    }
    // Start with u[0] = 0.
    u[0] = 0;
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (problem->BFS[i][j]==1) { // basic cell
                    if (u[i] != FLT_MAX && v[j] == FLT_MAX) {
                        v[j] = problem->cost[i][j] - u[i];
                        changed = 1;
                    }
                    if (v[j] != FLT_MAX && u[i] == FLT_MAX) {
                        u[i] = problem->cost[i][j] - v[j];
                        changed = 1;
                    }
                }
            }
        }
    }
}

// MODI algorithm implementation.
// Iteratively computes potentials, finds the most negative opportunity cost for nonbasic cells,
// and then adjusts the allocations along the corresponding closed loop until optimality is reached.
// Return the running time
double modiCPUSolve(TransportationProblem *problem) {
    printf("Running MODI Method for phase 2 optimization...\n");
    clock_t modi_start_time = clock();

    // epsilonAllocation(problem,1e-10);
    int m = problem->numSupply;
    int n = problem->numDemand;

    // Allocate arrays for the dual potentials.
    double *u = (double*)malloc(m * sizeof(double));
    double *v = (double*)malloc(n * sizeof(double));

    int optimal = 0;
    while (!optimal) {
        // Step 1: Compute potentials based on the current basic solution.
        computePotentials(problem, u, v, m, n);
        
        // Step 2: Compute opportunity costs (delta) for nonbasic cells.
        double bestDelta = 0.0;
        int best_i = -1, best_j = -1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (problem->BFS[i][j]==1)
                    continue; // Skip basic cells.
                double delta = problem->cost[i][j] - (u[i] + v[j]);
                if (delta < bestDelta || (delta <= bestDelta+1e-10 && (i<best_i||(i==best_i&& j<best_j)))) {
                    bestDelta = delta;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        
        // If no negative delta is found, the current solution is optimal.
        if (best_i == -1 || best_j == -1) {
            optimal = 1;
            break;
        }
        
        // Step 3: For the candidate cell (best_i, best_j), find a closed loop.
        int loop[m+n][2];
        int loop_length = 0;
        if (!findLoop(problem, m, n, best_i, best_j, loop, &loop_length)) {
            printf("No loop found for candidate cell (%d, %d) with reduced cost %f.\n", best_i, best_j, bestDelta);
            break;
        }
        
        // Step 4: Determine theta, the maximum feasible adjustment,
        // which is the minimum allocation among the cells in the loop with negative signs.
        double theta = FLT_MAX;
        int sign = -1; // Candidate cell is the plus position; subsequent positions alternate.
        for (int k = 1; k < loop_length; k++) {
            if (sign < 0) {
                int r = loop[k][0];
                int c = loop[k][1];
                if (problem->solution[r][c] < theta)
                    theta = problem->solution[r][c];
            }
            sign = -sign;
        }
        // printf("Candidate cell (%d, %d)\n", best_i, best_j);
        // printf("Improve the objective by %f, delta=%f, theta=%f\n", -bestDelta*theta, bestDelta, theta);


        // Step 5: Adjust allocations along the loop.
        sign = 1;
        for (int k = 0; k < loop_length; k++) {
            int r = loop[k][0];
            int c = loop[k][1];
            // printf("(%d,%d)->(%f, %d)  ",r,c,problem->solution[r][c],problem->BFS[r][c]);
            if (sign > 0)
                problem->solution[r][c] += theta;
            else{
                if(problem->solution[r][c]==theta)
                    problem->BFS[r][c]=0;
                problem->solution[r][c] -= theta;
            }
            sign = -sign;
        }
        // printf("\n");
        // Update BFS
        problem->BFS[best_i][best_j]=1;
        if(problem->solution[best_i][best_j]<1e-10)
            problem->solution[best_i][best_j]+=1e-10;
    }
    
    free(u);
    free(v);
    // Calculate and print the elapsed time.
    double elapsed_time = (double)(clock() - modi_start_time) / CLOCKS_PER_SEC;
    printf("CPU MODI solver completed in %f seconds.\n", elapsed_time);
    return elapsed_time;
}
