//Single-qubit gate operation can be simulated as many 2x2 matrix multiplications


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Compute single-qubit gate operation using Input array "in" and Gate "u".
 */
 
__global__ void
singQubitGate(const float *in, float *out, const float *u, int n, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	bool bq;
	
	if(i<numElements)
	{
		(i & (1<<n))?bq=1:bq=0;
		
		if(bq==0) out[i] = u[0]*in[i]           +  u[1]*in[i|(1<<n)];
		else      out[i] = u[2]*in[i&(~(1<<n))] +  u[3]*in[i];
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
	
	//File variables
	std::ifstream        f_in;             // File handler	
	std::vector<float> inp;
	float *u= (float *)malloc(4*sizeof(float));
	int numElements=0, n;
	
	f_in.open("input.txt", std::ifstream::in);
	
	for(int i=0; i<4; i++)
	{
		f_in >>u[i];
		//std::cout<<i<<" "<<u[i]<<std::endl;
	}
	    	
	float f;
	while (f_in >>f)
	{
		inp.push_back(f);
		//std::cout<<size<<" "<<inp[size]<<std::endl;
		numElements++;
	}
	
	f_in.close();
	
	n = (int)inp.back();
	inp.pop_back();
	numElements--;
	
	size_t size = numElements * sizeof(float);
	float *in= (float *)malloc(size);
	float *out= (float *)malloc(size);
	
	// Verify that allocations succeeded
    if (in == NULL || out == NULL || u == NULL )
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
	
    //Populate input array
    for(int i=0; i<numElements; i++) in[i]=inp[i];
	
    // Allocate the device input vector in
    float *d_in = NULL;
    err = cudaMalloc((void **)&d_in, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector out
    float *d_out = NULL;
    err = cudaMalloc((void **)&d_out, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_u = NULL;
    err = cudaMalloc((void **)&d_u, 4*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_u (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input d_in and d_u in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input array d_in from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_u, u, 4*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy qbit gate d_u from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the singQubitGate CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    singQubitGate<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_u, n, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch singQubitGate kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //cudaDeviceSynchronize();
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy out array d_out from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        printf(" %0.3f \n", out[i]);
    }
	
    //printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_in);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_out);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_u);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_u (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(in);
    free(out);
    free(u);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

