

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );

 

__global__ void dot( float *a, float *b, float *c ) {

    __shared__ float cache[threadsPerBlock];  //這裡是宣告chche 是 shared memory 裡面的變數

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int cacheIndex = threadIdx.x;  //因為 chche 的長度和threads 一樣，所以用threadIdx 即可



    float   temp = 0;

    while (tid < N) {

        temp += a[tid] * b[tid];

        tid += blockDim.x * gridDim.x;  //因為block*thread 的個數遠小於向量的長度，所以計算完一個片段之後要跳到下一段去計算

    }

    // set the cache values

    cache[cacheIndex] = temp;  //將計算完的結果存在cache 裡面

 

    // synchronize threads in this block

    __syncthreads();  //這一行會保證每一個thread 都執行了上面一行命令之後才會往下計算

 

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code

    int i = blockDim.x/2;

    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    } 

    // 上面這一段加法，是典型多工的加法，以往的CPU 加法就是從頭加到尾，因為只有一個CPU 來計算。
    //    當我們有很多個計算單元時，每一個計算單元都可以負責一個子集合，他們將這個子集合的元素加總之後，再把結果丟給CPU 來一個一個加。 


    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];

}


int main( void ) {

    float   *a, *b, c, *partial_c;

    float   *dev_a, *dev_b, *dev_partial_c;

 

    // allocate memory on the cpu side

    a = (float*)malloc( N*sizeof(float) );

    b = (float*)malloc( N*sizeof(float) );

    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );

 

    // allocate the memory on the GPU

    HANDLE_ERROR( cudaMalloc( (void**)&dev_a,

                              N*sizeof(float) ) );

    HANDLE_ERROR( cudaMalloc( (void**)&dev_b,

                              N*sizeof(float) ) );

    HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c,

                              blocksPerGrid*sizeof(float) ) );

 

    // fill in the host memory with data

    for (int i=0; i<N; i++) {

        a[i] = i;

        b[i] = i*2;

    }

    // copy the arrays 'a' and 'b' to the GPU

    HANDLE_ERROR( cudaMemcpy( dev_a, a, N*sizeof(float),

                              cudaMemcpyHostToDevice ) );

    HANDLE_ERROR( cudaMemcpy( dev_b, b, N*sizeof(float),

                              cudaMemcpyHostToDevice ) );

 

    dot<<<blocksPerGrid,threadsPerBlock>>>( dev_a, dev_b,

                                            dev_partial_c );

 

    // copy the array 'c' back from the GPU to the CPU

    HANDLE_ERROR( cudaMemcpy( partial_c, dev_partial_c,

                              blocksPerGrid*sizeof(float),

                              cudaMemcpyDeviceToHost ) );

 

    // finish up on the CPU side

    c = 0;

    for (int i=0; i<blocksPerGrid; i++) {

        c += partial_c[i];

    }

    #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)

    printf( "Does GPU value %.6g = %.6g?\n", c,

             2 * sum_squares( (float)(N - 1) ) );

 

    // free memory on the gpu side

    HANDLE_ERROR( cudaFree( dev_a ) );

    HANDLE_ERROR( cudaFree( dev_b ) );

    HANDLE_ERROR( cudaFree( dev_partial_c ) );

 

    // free memory on the cpu side

    free( a );

    free( b );

    free( partial_c );

}