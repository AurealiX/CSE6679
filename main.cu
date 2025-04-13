#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "transportation.h"
#include "util.h"
#include "cpu_vam.h"
#include "cpu_modi.h"
#include "gpu_modi.h"
#include "cpu_lcm.h"
#include "gpu_lcm.h"
#include "cpu_ssm.h"
#include "gpu_vam.h"
#include "gpu_ssm.h"

// Clone instance helper
TransportationProblem* cloneTransportationProblem(TransportationProblem* original) {
    int m = original->numSupply, n = original->numDemand;
    TransportationProblem* copy = createTransportationProblem(m, n);
    memcpy(copy->supply, original->supply, m*sizeof(double));
    memcpy(copy->demand, original->demand, n*sizeof(double));
    for (int i = 0; i < m; i++) {
        memcpy(copy->cost[i], original->cost[i], n*sizeof(double));
        memcpy(copy->solution[i], original->solution[i], n*sizeof(double));
        memcpy(copy->BFS[i], original->BFS[i], n*sizeof(int));
    }
    return copy;
}

// Helper function to solve and report results
double solveAndReport(TransportationProblem* inst, const char* method, double (*solveFunc)(TransportationProblem*)) {
    double time = solveFunc(inst);
    printf("%s: feas=%s, obj=%f\n", 
           method, 
           isFeasible(inst) ? "true" : "false", 
           getObjectiveValue(inst));
    return time;
}

int main() {
    int num_supply = 100, num_demand = 57, max_cost = 10, seed = 1;
    printf("Enter seed: ");
    scanf("%d", &seed);
    printf("Enter supply nodes: ");
    scanf("%d", &num_supply);
    printf("Enter demand nodes: ");
    scanf("%d", &num_demand);
    int max_supply = num_supply + num_demand;
    
    // New: 4-digit binary mode (phase1_alg, phase1_hw, phase2_alg, phase2_hw)
    // phase1_alg: 0=VAM, 1=LCM
    // phase1_hw: 0=CPU, 1=GPU
    // phase2_alg: 0=MODI, 1=SSM
    // phase2_hw: 0=CPU, 1=GPU
    char mode[5];
    printf("Enter 4-digit binary mode:\n");
    printf("  First digit: 0=VAM, 1=LCM for phase 1\n");
    printf("  Second digit: 0=CPU, 1=GPU for phase 1\n");
    printf("  Third digit: 0=MODI, 1=SSM for phase 2\n");
    printf("  Fourth digit: 0=CPU, 1=GPU for phase 2\n");
    printf("Mode: ");
    scanf("%4s", mode);
    
    double phase1Time = 0, phase2Time = 0;
    TransportationProblem* inst;
    const char* phase1_method;
    const char* phase2_method;
    double (*phase1_func)(TransportationProblem*);
    double (*phase2_func)(TransportationProblem*);
    
    // Determine phase 1 algorithm and hardware
    if (mode[0] == '0') { // VAM
        if (mode[1] == '0') { // CPU
            phase1_method = "VAM (CPU)";
            phase1_func = vamCPUSolve;
        } else { // GPU
            phase1_method = "VAM (GPU)";
            phase1_func = gpuVamSolve;
        }
    } else { // LCM
        if (mode[1] == '0') { // CPU
            phase1_method = "LCM (CPU)";
            phase1_func = lcmCPUSolve;
        } else { // GPU
            phase1_method = "LCM (GPU)";
            phase1_func = lcmGPUSolve;
        }
    }
    
    // Determine phase 2 algorithm and hardware
    if (mode[2] == '0') { // MODI
        if (mode[3] == '0') { // CPU
            phase2_method = "MODI (CPU)";
            phase2_func = modiCPUSolve;
        } else { // GPU
            phase2_method = "MODI (GPU)";
            phase2_func = modiGPUSolve;
        }
    } else { // SSM
        if (mode[3] == '0') { // CPU
            phase2_method = "SSM (CPU)";
            phase2_func = ssmCPUSolve;
        } else { // GPU
            phase2_method = "SSM (GPU)";
            phase2_func = ssmGPUSolve;
        }
    }
    
    // Phase 1: Initial solution
    inst = createTransportationProblem(num_supply, num_demand);
    generateRandomInstance(inst, num_supply, num_demand, max_supply, 0, max_cost, seed);
    printf("Phase 1: %s\n", phase1_method);
    phase1Time = solveAndReport(inst, phase1_method, phase1_func);
    
    // Phase 2: Optimize from initial solution
    printf("Phase 2: %s\n", phase2_method);
    phase2Time = solveAndReport(inst, phase2_method, phase2_func);
    
    // Final times
    printf("Time: Phase 1: %f sec, Phase 2: %f sec\n", phase1Time, phase2Time);
    
    // Clean up
    destroyTransportationProblem(inst);
    
    return 0;
}