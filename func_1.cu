#include <stdio.h>    
#include <stdlib.h>   
#include <cuda_runtime.h>  
 
#define SIZE 1024
 
__global__ void histo_kernel(int size, unsigned int *histo)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size)
	{
		//*histo+=i;
		atomicAdd(histo, i);
	}
}
 

