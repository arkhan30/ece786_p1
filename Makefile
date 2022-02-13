# Location of the CUDA Toolkit
NVCC := $(CUDA_PATH)/bin/nvcc
CCFLAGS := -O2

EXTRA_NVCCFLAGS := --cudart=shared
build: quamsimV1

quamsimV1.o:quamsimV1.cu
	$(NVCC) $(INCLUDES) $(CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

quamsimV1: quamsimV1.o
$(NVCC) $(LDFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./quamsimV1
