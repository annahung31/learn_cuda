




__global__ void add(int *a, int *b, int *c){
    *c = *a + *b;
}

int main(void)(
    int ha=1,hb=2,hc;
    add<<<1,1>>>(&ha, &hb, &hc);
    printf("c=%d\n", hc);
    return 0;
)


// This doesn'n work! Because int ha,hb,hc are in the host mem, which 
// cannot be used by GPU(device).
// NEED to allocate variables in "device mem" as following:




int main(void){
    int a=1, b=2, c;
    int *d_a, *d_b, *d_c;  // "*" means pointer.

    // allocate space for device variables
    // "**" means pointer of pointer.
    cudaMalloc((void **)&d_a, sizeof(int));
    cudaMalloc((void **)&d_b, sizeof(int));
    cudaMalloc((void **)&d_c, sizeof(int));

    // Copy input to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel on GPU
    add<<<1,1>>>(d_a, d_b, d_c);

    // After finishing operation, copy result from GPU back to CPU.
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free the mem if no longer use it.
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;

}
