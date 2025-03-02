#include "transportation.h"
#include <stdlib.h>
#include "util.h"
#include <time.h>

TransportationProblem* createTransportationProblem(int numSupply, int numDemand) {
    TransportationProblem* problem = (TransportationProblem*)malloc(sizeof(TransportationProblem));
    problem->numSupply = numSupply;
    problem->numDemand = numDemand;
    problem->supply = (double*)malloc(numSupply * sizeof(double));
    problem->demand = (double*)malloc(numDemand * sizeof(double));
    problem->cost = create_double_matrix(numSupply,numDemand);
    // Initialize the solution matrix to zero.
    problem->solution = create_double_matrix(numSupply,numDemand);
    matrix_set_all(problem->solution,numSupply,numDemand,0);
    problem->BFS = create_int_matrix(numSupply,numDemand);
    matrix_set_all(problem->BFS,numSupply,numDemand,0);
    return problem;
}

void destroyTransportationProblem(TransportationProblem *problem) {
    if (problem) {
        free(problem->supply);
        free(problem->demand);
        // Use the number of supply nodes stored in the structure
        free_matrix(problem->cost, problem->numSupply);
        free_matrix(problem->solution, problem->numSupply);
        free_matrix(problem->BFS, problem->numSupply);
        free(problem);
    }
}

// Populates the TransportationProblem with random data.
void generateRandomInstance(TransportationProblem *problem, int numSupply, int numDemand, double totalSupply, double minCost, double maxCost, int seed) {
    // Seed the random generator.
    srand((unsigned int)seed);

    double total=0;
    double* a=(double*) malloc( sizeof(double) * numSupply);
    for (int i = 0; i < numSupply; i++) {
        a[i] = randomDouble(10.0f, 100.0f);
        total+=a[i];
    }
    for (int i = 0; i < numSupply; i++) {
        problem->supply[i] = a[i]/total*totalSupply;
    }
    free(a);

    total=0;
    double* b=(double*) malloc(sizeof(double) * numDemand);
    for (int j = 0; j < numDemand; j++) {
        b[j] = randomDouble(10.0f, 100.0f);
        total+=b[j];
    }
    for (int j = 0; j < numDemand; j++) {
        problem->demand[j] = b[j]/total*totalSupply;
    }
    free(b);

    for (int i = 0; i < numSupply; i++) {
        for (int j = 0; j < numDemand; j++) {
            problem->cost[i][j] = randomDouble(minCost, maxCost);
        }
    }
}

// Cauculate the current objective value. 
double getObjectiveValue(TransportationProblem* problem){
    double total=0;
    int m=problem->numSupply;
    int n=problem->numDemand;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++)
            total+=problem->cost[i][j]*problem->solution[i][j];
    }
    return total;
}

// Check feasibility
bool isFeasible(TransportationProblem* problem){
    int m=problem->numSupply;
    int n=problem->numDemand;

    for(int i=0;i<m;i++){
        double row_sum=0;
        for(int j=0;j<n;j++) 
            row_sum+=problem->solution[i][j];
        double diff=row_sum-problem->supply[i];
        if(diff>1e-5||diff<-1e-5) 
            return false;
    }

    for(int j=0;j<n;j++){
        double col_sum=0;
        for(int i=0;i<m;i++) 
            col_sum+=problem->solution[i][j];
        double diff=col_sum-problem->demand[j];
        if(diff>1e-5||diff<-1e-5) 
            return false;
    }

    return true;
}
