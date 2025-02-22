#ifndef TRANSPORTATION_H
#define TRANSPORTATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

// Data structure for a transportation problem.
typedef struct {
    int numSupply;
    int numDemand;
    double *supply;    // Supply vector.
    double *demand;    // Demand vector.
    double **cost;      // Cost matrix.
    double **solution;  // Allocation matrix.
    int **BFS; // indicate which entries correspond to basic feasible solution
} TransportationProblem;

// Allocates and initializes a transportation problem.
TransportationProblem* createTransportationProblem(int numSupply, int numDemand);

// Frees a transportation problem instance.
void destroyTransportationProblem(TransportationProblem *problem);

//Generate a random instance
void generateRandomInstance(TransportationProblem *problem, int numSupply, int numDemand, double totalSupply, double minCost, double maxCost, int seed);

// Cauculate the current objective value. 
double getObjectiveValue(TransportationProblem* problem);

// Check feasibility
bool isFeasible(TransportationProblem* problem);


#ifdef __cplusplus
}
#endif

#endif // TRANSPORTATION_H
