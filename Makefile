# NVIDIA C++ compiler and flags.
NVCC        = nvcc

LDFLAGS	= -L$(CUDA_ROOT)/lib64 -lcudart

# Hardware-specific flags for NVIDIA GPU generations. Add any/all.
#GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
#GENCODE_SM35    := -gencode arch=compute_37,code=sm_37
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM52    := -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75    := -gencode arch=compute_75,code=\"sm_75,compute_75\"

GENCODE_FLAGS   := $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM37)             \
$(GENCODE_SM50) $(GENCODE_SM52) $(GENCODE_SM60) $(GENCODE_SM70) $(GENCODE_SM75)

#name executable
EXE	        = TPSolver
OBJ = main.o transportation.o util.o cpu_vam.o cpu_modi.o cpu_lcm.o gpu_lcm.o gpu_vam.o gpu_modi.o cpu_ssm.o gpu_ssm.o

#build up flags and targets
NVCCFLAGS=-O0 $(GENCODE_FLAGS) -Xcompiler -march=native

NVCCFLAGS+= -c --std=c++11

# Add these flags:
NVCCFLAGS += -G -g -Xcompiler -fdebug-prefix-map=$(CURDIR)=.

NVCCFLAGS+= -DWITH_CUBLAS -I $(CUDA_ROOT)/include
LDFLAGS+= -lcublas -L $(CUDA_ROOT)/lib64 -Xcompiler \"-Wl,-rpath,$(CUDA_ROOT)/lib64\"

all: $(EXE)

# Updated compilation rules for your files
main.o: main.cu transportation.h util.h cpu_vam.h cpu_modi.h
	$(NVCC) -c -o $@ main.cu $(NVCCFLAGS)

transportation.o: transportation.cu transportation.h
	$(NVCC) -c -o $@ transportation.cu $(NVCCFLAGS)

util.o: util.cu util.h
	$(NVCC) -c -o $@ util.cu $(NVCCFLAGS)

cpu_vam.o: cpu_vam.cu cpu_vam.h 
	$(NVCC) -c -o $@ cpu_vam.cu $(NVCCFLAGS)

cpu_modi.o: cpu_modi.cu cpu_modi.h 
	$(NVCC) -c -o $@ cpu_modi.cu $(NVCCFLAGS)

cpu_lcm.o: cpu_lcm.cu cpu_lcm.h
	$(NVCC) -c -o $@ cpu_lcm.cu $(NVCCFLAGS)

gpu_lcm.o: gpu_lcm.cu gpu_lcm.h
	$(NVCC) -c -o $@ gpu_lcm.cu $(NVCCFLAGS)

gpu_vam.o: gpu_vam.cu gpu_vam.h
	$(NVCC) -c -o $@ gpu_vam.cu $(NVCCFLAGS)

gpu_modi.o: gpu_modi.cu gpu_modi.h
	$(NVCC) -c -o $@ gpu_modi.cu $(NVCCFLAGS)

cpu_ssm.o: cpu_ssm.cu cpu_ssm.h
	$(NVCC) -c -o $@ cpu_ssm.cu $(NVCCFLAGS)

gpu_ssm.o: gpu_ssm.cu gpu_ssm.h
	$(NVCC) -c -o $@ gpu_ssm.cu $(NVCCFLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) $(LDFLAGS) -o $(EXE)

clean:
	/bin/rm -rf *.o $(EXE)
