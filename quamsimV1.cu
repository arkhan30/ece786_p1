/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

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
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
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
	float *u= (float *)malloc(128*sizeof(float));
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
	
	//memcpy(&in, &inp, size);
	for(int i=0; i<numElements; i++) in[i]=inp[i];
	

    // Allocate the device input vector in
    float *d_in = NULL;
    err = cudaMalloc((void **)&d_in, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector out
    float *d_out = NULL;
    err = cudaMalloc((void **)&d_out, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_u = NULL;
    err = cudaMalloc((void **)&d_u, 128*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    //printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy input array from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_u, u, 128*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy qbit gate from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    singQubitGate<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_u, n, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //cudaDeviceSynchronize();
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    //printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy out array from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
   
    
	/*int temp_0, temp_1;
	int mask = pow(2, n+1) -1;
	
	for(int i=0; i<numElements/2; i++)
	{		
        temp_0 = ((i<<1) & ~mask);
	    temp_0 = ((i & mask)|temp_0);	
		temp_0 = temp_0 & (~(1<<n));
		
	
	    temp_1 = ((i<<1) & ~mask);
	    temp_1 = ((i & mask)|temp_1);
		temp_1 = temp_1 | (1<<n);
		
		//std::cout<<"temp_0 "<<temp_0<<std::endl;
		//std::cout<<"temp_1 "<<temp_1<<std::endl;
		
	
        out[temp_0] = u[0]*in[temp_0]+ u[1]*in[temp_1];
		//std::cout<<"in[temp_0] "<<in[temp_0]<<" in[temp_1] "<<in[temp_1]<<std::endl;
		//std::cout<<"u[0]= "<<u[0]<<" u[1]= "<<u[1]<<" out[temp_0] "<<out[temp_0]<<std::endl;
	    out[temp_1] = u[2]*in[temp_0]+ u[3]*in[temp_1];
		
		std::cout<<"in[temp_0] "<<in[temp_0]<<std::endl;
		std::cout<<"in[temp_1] "<<in[temp_1]<<std::endl;
		std::cout<<"out[temp_0] "<<out[temp_0]<<std::endl;
		std::cout<<"out[temp_1] "<<out[temp_1]<<std::endl;
	}*/
	
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

