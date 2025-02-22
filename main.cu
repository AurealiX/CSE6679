#include <stdio.h>
#include "transportation.h"
#include "util.h"
#include "cpu_vam.h"
#include "cpu_modi.h"

int main() {
    // Parameters for the random instance
    int num_supply = 100;
    int num_demand = 57;
    int max_cost = 10;   // Maximum cost value for each cell
    
    int seed=1;
    printf("Enter seed: ");
    scanf("%d", &seed);  

    printf("Enter number of supply nodes: ");
    scanf("%d", &num_supply); 

    printf("Enter number of demand nodes: ");
    scanf("%d", &num_demand); 

    int max_supply = num_supply+num_demand; // Maximum supply value per source

    // Generate a random test instance
    TransportationProblem* problem= createTransportationProblem(num_supply,num_demand);
    generateRandomInstance(problem, num_supply, num_demand, max_supply, 0, max_cost,1);

    printf("Find initial solution in: %f \n", vamCPUSolve(problem));

    // printDoubleMatrix(problem->solution,num_supply,num_demand);
    // printIntMatrix(problem->BFS,num_supply,num_demand);

    printf("Initial solution feasibility is %s, the objective value is %f \n", isFeasible(problem)? "true": "false", getObjectiveValue(problem));

    printf("Phase 2 optimization completed in: %f \n", modiCPUSolve(problem));

    printf("Final solution feasibility is %s, the objective value is %f \n", isFeasible(problem)? "true": "false", getObjectiveValue(problem));

    return 0;
}
